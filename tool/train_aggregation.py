import os
import random
import time
import cv2
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist

import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(1, os.path.abspath('..'))

from util import dataset, transform, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, find_free_port

from model.pspnet_agg import PSPNetAggregation

from sklearn.metrics import roc_auc_score

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

value_scale = 255
mean = [0.485, 0.456, 0.406]
mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
std = [item * value_scale for item in std]

data_root = "dataset/ade20k"
train_list = "dataset/ade20k/list/training.txt"
valid_list = "dataset/ade20k/list/validation.txt"
batch_size = 16
epochs = 15
n_classes = 150

def mean_auc(classification_labels, classification_predictions):
    """
    labels/predictions are lists of tensors [[n, num_classes, size, size] with sizes 1, 2, 3, 6]
    """
    class_aucs = [0] * n_classes
    class_counts = [0] * n_classes
    # loop over each classification head
    for scale_idx in range (len(classification_labels)):
        y_true, y_pred = classification_labels[scale_idx], classification_predictions[scale_idx]
        y_true, y_pred = y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
        # loop over n_classes
        for c in range(n_classes):
            # loop over spatial predictions
            for x in range(y_true.shape[3]):
                for y in range(y_true.shape[3]):
                    if len(np.unique(y_true[:,c,x,y])) == 2:
                        bin_vec_true = y_true[:,c,x,y]
                        bin_vec_pred = y_pred[:,c,x,y]
                        auc = roc_auc_score(bin_vec_true, bin_vec_pred)
                        class_aucs[c] += auc
                        class_counts[c] += 1
    # take care of classes which were not present in the batch
    class_mean_aucs = []
    for c in range(n_classes):
        if class_counts[c] > 0:
            class_mean_aucs.append(class_aucs[c]/class_counts[c])
    # return mean auc of classes present in the batch
    mean_auc_batch = np.mean(class_mean_aucs)
    return mean_auc_batch
    

def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = epochs * len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input, classifications = input
        classifications = [ct.float().cuda(non_blocking=True) for ct in classifications]
        seg_target = target
        # seg_target, class_target = target
        seg_target = seg_target.cuda(non_blocking=True)
        # class_target = [ct.float().cuda(non_blocking=True) for ct in class_target]
        input = input.cuda(non_blocking=True)
        segmentation, main_loss = model(x=input, y=seg_target, classifications=classifications)
        # segmentation, classification, main_loss, aux_loss, classification_loss = model(input, target)
        main_loss = torch.mean(main_loss)
        loss = main_loss

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        n = input.size(0)

        intersection, union, target = intersectionAndUnionGPU(segmentation, seg_target, 150, 255)

        # auc = mean_auc(class_target, classification)

        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(main_loss.item(), n)
        # aux_loss_meter.update(aux_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(1e-2, current_iter, max_iter, power=0.9)
        for index in range(0, 1):
            optimizer.param_groups[index]['lr'] = current_lr
        # for index in range(args.index_split, len(optimizer.param_groups)):
        #     optimizer.param_groups[index]['lr'] = current_lr * 10
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        # print('Epoch: [{}/{}][{}/{}] '
        #         'Auc: {}'.format(epoch+1, epochs, i + 1, len(train_loader), auc))

        # full printout for segmentation
        print('Epoch: [{}/{}][{}/{}] '
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                'Remain {remain_time} '
                'MainLoss {main_loss_meter.val:.4f} '
                'Loss {loss_meter.val:.4f} '
                'Accuracy {accuracy:.4f}.'.format(epoch+1, epochs, i + 1, len(train_loader),
                                                    batch_time=batch_time,
                                                    data_time=data_time,
                                                    remain_time=remain_time,
                                                    main_loss_meter=main_loss_meter,
                                                    loss_meter=loss_meter,
                                                    accuracy=accuracy))
    
        print('loss_train_batch', main_loss_meter.val, current_iter)
        print('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
        print('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
        print('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    print('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, epochs, mIoU, mAcc, allAcc))
    return main_loss_meter.avg, mIoU, mAcc, allAcc

def validate(model):
    val_transform = transform.Compose([
        transform.Crop([473, 473], crop_type='center', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    val_data = dataset.SemData(split='val', data_root=data_root, data_list=valid_list, transform=val_transform, classification_heads_x=False, classification_heads_y=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, sampler=None)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    intersection_meter2 = AverageMeter()
    union_meter2 = AverageMeter()
    target_meter2 = AverageMeter()

    model.eval()
    end = time.time()
    ious = []
    accs = []
    ious2 = []
    accs2 = []

    logit_1 = []
    logit_2 = []
    prob_1 = []
    dist = []
    for i, (input, target) in enumerate(val_loader):
        seg_target, dist_target = target
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        seg_target = seg_target.cuda(non_blocking=True)
        dist_target = dist_target[0]
        dist_target = dist_target.cuda(non_blocking=True)
        output, output_alt = model(input, distributions=dist_target)

        # seg_target_mask = torch.where(seg_target == 255, output.max(1)[1], seg_target)
        # seg_onehot = nn.functional.one_hot(seg_target_mask, num_classes=150).float().permute(0, 3, 1, 2)
        # pixel_dist = nn.AdaptiveAvgPool2d((60, 60))(seg_onehot)
        # dist.append(pixel_dist)

        # logit_1.append(logits[-1])
        # logit_2.append(nn.ReLU()(logits[-1]))
        # prob_1.append(nn.Sigmoid()(logits[-1]))

        n = input.size(0)

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, seg_target, 150, 255)
        
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        ious.append(intersection / (union + 1e-10))
        accs.append(intersection / (target + 1e-10))

        output_alt = output_alt.max(1)[1]
        intersection2, union2, target2 = intersectionAndUnionGPU(output_alt, seg_target, 150, 255)
        
        intersection2, union2, target2 = intersection2.cpu().numpy(), union2.cpu().numpy(), target2.cpu().numpy()
        intersection_meter2.update(intersection2), union_meter2.update(union2), target_meter2.update(target2)

        ious2.append(intersection2 / (union2 + 1e-10))
        accs2.append(intersection2 / (target2 + 1e-10))

        batch_time.update(time.time() - end)
        end = time.time()

    # logit_1 = torch.stack([x.cpu() for x in logit_1]) 
    # logit_2 =torch.stack([x.cpu() for x in logit_2]) 
    # prob_1 = torch.stack([x.cpu() for x in prob_1]) 
    # dist = torch.stack([x.cpu() for x in dist]) 

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    print('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))

    iou_class2 = intersection_meter2.sum / (union_meter2.sum + 1e-10)
    accuracy_class2 = intersection_meter2.sum / (target_meter2.sum + 1e-10)
    mIoU2 = np.mean(iou_class2)
    mAcc2 = np.mean(accuracy_class2)
    allAcc2 = sum(intersection_meter2.sum) / (sum(target_meter2.sum) + 1e-10)
    print('Val result2: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU2, mAcc2, allAcc2))
    return mIoU, mAcc, allAcc

def main(model):
    learning_rate = 1e-3
    # params_list = [dict(params=model.channel_weights.parameters(), lr=learning_rate)]
    # optimizer = torch.optim.SGD(params_list, lr=1e-2, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(lr=learning_rate, weight_decay=1e-4)
    train_epochs = [ ]
    val_epochs = [ ]
    train_transform = transform.Compose([
    transform.RandScale([0.5, 2.0]),
    transform.RandRotate([-10, 10], padding=mean, ignore_label=255),
    transform.RandomGaussianBlur(),
    transform.RandomHorizontalFlip(),
    transform.Crop([473, 473], crop_type='rand', padding=mean, ignore_label=255),
    transform.ToTensor(),
    transform.Normalize(mean=mean, std=std)])
    train_data = dataset.SemData(split='train', data_root=data_root, data_list=train_list, transform=train_transform, classification_heads_x=True, classification_heads_y=False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, sampler=None, drop_last=True)
    for epoch in range(0, epochs):
        # mIoU, mAcc, allAcc = validate(model)
        # val_epochs.append((mIoU, mAcc, allAcc))
        _, t_mIoU, t_mAcc, t_allAcc = train(train_loader, model, optimizer, epoch)
        train_epochs.append((t_mIoU, t_mAcc, t_allAcc))
        
    # torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, f"exp/ade20k/pspnet50/model/asdfasdf.pth")
    return val_epochs
    # validate(model)

if __name__ == "__main__":
    # bottleneck 1x1 -> n_classes 3x3, no bias
    # Val result: mIoU/mAcc/allAcc 0.4216/0.5595/0.7897.

    # 3x3 only with bias (nearest upsampling)
    # Train 0.7559/0.8216/0.9233
    # Val 0.3218/0.4034/0.7345
    # 3x3 only without bias (nearest upsampling)
    # Train 0.7750/0.8395/0.9283
    # Val 0.3185/0.4113/0.7243

    # 1x1 linear bottleneck 3x3 with bias
    # Train 0.7384/0.8142/0.9183
    # Val 0.4223, 0.5312, 0.7974

    # 1x1 linear bottleneck 3x3 without bias
    # Train 0.7327/0.8095/0.9163
    # Val 0.4156/0.5236/0.7957


    # conv combination
    # Train 6505/0.7651/0.8962
    # Val 0.4134/0.5269/0.7936
    # 0.4183/0.5376/0.7945
    model_conv = PSPNetAggregation(pspnet_weights="distributions_ae.pth").to("cuda")
    val_hist_conv = main(model_conv)
    print(val_hist_conv)

    # Val result: mIoU/mAcc/allAcc 0.4007/0.4888/0.7865.
    # model_mlp = PSPNetAggregation(pspnet_weights="exp/ade20k/pspnet50/model/classification.pth", agg="mlp").to("cuda")
    # val_hist_mlp = main(model_mlp, agg_type="mlp")
    # print(val_hist_conv)
    # print(val_hist_mlp)