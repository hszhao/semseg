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

import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(1, os.path.abspath('..'))

from util import dataset, transform, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, find_free_port, colorize
from util.classification_utils import extract_mask_distributions, extract_adjusted_distribution
#from tool.test import multiscale_prediction
#from tool.visualization import dist_prediction_visualization, cam

from model.pspnet_context import PSPNetContext
from model.pspnet_ms_context import PyramidContextNetwork
# from model.encnet import EncNet
# from model.fcn import FCN

import seaborn as sns

from sklearn.metrics import roc_auc_score

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

value_scale = 255
mean = [0.485, 0.456, 0.406]
mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
std = [item * value_scale for item in std]

data_root = "dataset/ade20k"
# parition 10% of training set for hyperparameter tuning, to report real results on validation set.
train_list = "dataset/ade20k/list/training_alt.txt"
valid_list = "dataset/ade20k/list/validation_alt.txt"
test_list = "dataset/ade20k/list/validation.txt"
batch_size = 16
epochs = 10
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
            for x in range(y_true.shape[-1]):
                for y in range(y_true.shape[-1]):
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

def distribution_distance(distribution_labels, distribution_predictions):
    distances = []
    for scale in range(len(distribution_labels)):
        y_true, y_pred = distribution_labels[scale], distribution_predictions[scale]
        y_true, y_pred = y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
        distance = np.sum(np.abs(y_true - y_pred))
        distances.append(distance)
    return distances

# def debug(data_loader, model, i=3):
#     """
#     run visualization fn (e.g. CAM) on ith input of data loader
#     """
#     model.eval()
#     for idx, (input, target) in enumerate(data_loader):
#         input = input.cuda(non_blocking=True)
#         seg_target, class_target = target
#         seg_target = seg_target.cuda(non_blocking=True)
#         if idx == i:
#             cam(model, input, seg_target)
#             break

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
        #input, classifications = input
        # classifications = [ct.float().cuda(non_blocking=True) for ct in classifications]
        # seg_target = target
        input = input.cuda(non_blocking=True)
        
        seg_target, context_target = target
        seg_target = seg_target.cuda(non_blocking=True)
        context_target = [ct.float().cuda(non_blocking=True) for ct in context_target]
        context_target = context_target[0] # only care single scale

        segmentation, loss = model(input, [seg_target, context_target])
        loss = torch.mean(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = input.size(0)

        intersection, union, target = intersectionAndUnionGPU(segmentation, seg_target, 150, 255)

        # auc = mean_auc(context_target, classifications)

        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(loss.item(), n)
        # aux_loss_meter.update(aux_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + i + 1
        # current_lr = poly_learning_rate(1e-2, current_iter, max_iter, power=0.9)
        # NEW_MODULES = 7
        # for index in range(0, NEW_MODULES):
        #     optimizer.param_groups[index]['lr'] = current_lr * 10
        # for index in range(NEW_MODULES, len(optimizer.param_groups)):
        #     optimizer.param_groups[index]['lr'] = current_lr * 10
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        # full printout for segmentation
        print('Epoch: [{}/{}][{}/{}] '
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                'Remain {remain_time} '
                'MainLoss {main_loss_meter.val:.4f} '
                'Loss {loss_meter.val:.4f} '
                'Accuracy {accuracy:.4f}. '
                'mauc {auc:.4f}'.format(epoch+1, epochs, i + 1, len(train_loader),
                                                    batch_time=batch_time,
                                                    data_time=data_time,
                                                    remain_time=remain_time,
                                                    main_loss_meter=main_loss_meter,
                                                    loss_meter=loss_meter,
                                                    accuracy=accuracy,
                                                    auc=0))
    
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
    return main_loss_meter.avg, np.round(mIoU, 4), np.round(mAcc, 4), np.round(allAcc, 4)

def validate(model, data_list=valid_list):
    val_transform = transform.Compose([
        transform.Crop([473, 473], crop_type='center', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    val_data = dataset.SemData(split='val', data_root=data_root, data_list=data_list, transform=val_transform, context_x=False, context_y=True, context_type="distribution")
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, sampler=None)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    intersection_meter2 = AverageMeter()
    union_meter2 = AverageMeter()
    target_meter2 = AverageMeter()
    intersection_meter3 = AverageMeter()
    union_meter3 = AverageMeter()
    target_meter3 = AverageMeter()
    loss_meter = AverageMeter()

    model.eval()
    end = time.time()
    ious = []
    accs = []
    ious2 = []
    accs2 = []
    ious3 = []
    accs3 = []
    
    aucs = []
    for i, (input, target) in enumerate(val_loader):
        # seg_target = target
        seg_target, context_target = target
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        seg_target = seg_target.cuda(non_blocking=True)
        context_target = [ct.float().cuda(non_blocking=True) for ct in context_target]
        context_target = context_target[0]  # only look at global scale for now

        _, output, loss = model(input, [seg_target, context_target])

        n = input.size(0)
        loss_meter.update(loss.item(), n)

        # auc = mean_auc(context_target, classifications)
        # aucs.append(auc)

        # baseline model
        output = output.max(1)[1]
        # updated_context = extract_mask_distributions(seg_target.cpu(), top_k=5, predicted_mask=output.cpu()) # top k from predicted mask
        # updated_context = extract_adjusted_distribution(seg_target, output)  # without void
        
        intersection, union, target = intersectionAndUnionGPU(output, seg_target, 150, 255)
        
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        ious.append(intersection / (union + 1e-10))
        accs.append(intersection / (target + 1e-10))
        # residuals.append(residual.squeeze())

        # context model
        # output_alt = output_alt.max(1)[1] #torch.from_numpy(np.expand_dims(output_alt, 0)).to("cuda").max(3)[1]
        # intersection2, union2, target2 = intersectionAndUnionGPU(output_alt, seg_target, 150, 255)
        
        # intersection2, union2, target2 = intersection2.cpu().numpy(), union2.cpu().numpy(), target2.cpu().numpy()
        # intersection_meter2.update(intersection2), union_meter2.update(union2), target_meter2.update(target2)

        # ious2.append(intersection2 / (union2 + 1e-10))
        # accs2.append(intersection2 / (target2 + 1e-10))

        # context + corrections (novoid / top_k prediction experiment)
        # updated_context = np.asarray(updated_context).reshape(context_target.shape)

        # updated_context = torch.from_numpy(updated_context).cuda(non_blocking=True)
        # _, output_alt_exp = model(input, distributions=updated_context)

        # aggregation + novoid PSPNet
        # output_alt_exp = output_alt_exp.max(1)[1]
        # intersection3, union3, target3 = intersectionAndUnionGPU(output_alt_exp, seg_target, 150, 255)
        
        # intersection3, union3, target3 = intersection3.cpu().numpy(), union3.cpu().numpy(), target3.cpu().numpy()
        # intersection_meter3.update(intersection3), union_meter3.update(union3), target_meter3.update(target3)

        # ious3.append(intersection3 / (union3 + 1e-10))
        # accs3.append(intersection3 / (target3 + 1e-10))

        batch_time.update(time.time() - end)
        end = time.time()
        # print(f"{i+1}/{len(val_loader)}")

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    print('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))

    # print("Mean auc", np.mean(auc))

    # iou_class2 = intersection_meter2.sum / (union_meter2.sum + 1e-10)
    # accuracy_class2 = intersection_meter2.sum / (target_meter2.sum + 1e-10)
    # mIoU2 = np.mean(iou_class2)
    # mAcc2 = np.mean(accuracy_class2)
    # allAcc2 = sum(intersection_meter2.sum) / (sum(target_meter2.sum) + 1e-10)
    # print('Val result (+aggregation): mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU2, mAcc2, allAcc2))

    # iou_class3 = intersection_meter3.sum / (union_meter3.sum + 1e-10)
    # accuracy_class3 = intersection_meter3.sum / (target_meter3.sum + 1e-10)
    # mIoU3 = np.mean(iou_class3)
    # mAcc3 = np.mean(accuracy_class3)
    # allAcc3 = sum(intersection_meter3.sum) / (sum(target_meter3.sum) + 1e-10)
    # print('Val result (+novoid+aggregation): mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU3, mAcc3, allAcc3))
    return loss_meter.avg, np.round(mIoU, 4), np.round(mAcc, 4), np.round(allAcc, 4)

def main(model, dist_dim="all", top_k=150, decay=0):
    # define optimizer
    learning_rate = 2e-3
    modules_new = [model.combo.layer1, model.combo.layer2]
    params_list = []
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=learning_rate))
    optimizer = torch.optim.Adam(params_list, lr=learning_rate, weight_decay=decay)
    
    # viz_transform = transform.Compose([
    #     transform.RandScale([1.1, 1.2  ]),
    #     transform.Crop([473, 473], crop_type='center', padding=mean, ignore_label=255),
    #     transform.ToTensor(),
    #     transform.Normalize(mean=mean, std=std)
    # ])
    # for CAM and other visualization
    # viz_data = dataset.SemData(split='val', data_root=data_root, data_list=valid_list, transform=viz_transform, context_x=False, context_y=True, context_type="distribution")
    # viz_loader = torch.utils.data.DataLoader(viz_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, sampler=None, drop_last=True)
    # debug(viz_loader, model)

    train_transform = transform.Compose([
        transform.RandScale([0.5, 2.0]),
        transform.RandRotate([-10, 10], padding=mean, ignore_label=255),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([473, 473], crop_type='rand', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

    train_data = dataset.SemData(split='train', data_root=data_root, data_list=train_list, transform=train_transform, context_x=False, context_y=True, context_type="distribution")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, sampler=None, drop_last=True)

    train_epochs = [ ]
    val_epochs = [ ]
    test_epochs = [ ]
    for epoch in range(0, epochs):
        print(">>> BEGIN TRAIN EPOCH <<<")
        loss, t_mIoU, t_mAcc, t_allAcc = train(train_loader, model, optimizer, epoch)
        train_epochs.append((loss, t_mIoU, t_allAcc))
        print(">>> COMPUTING VALIDATION ERROR <<<")
        val_loss, val_mIoU, val_mAcc, val_allAcc = validate(model, data_list=valid_list)
        val_score = (val_mIoU + val_allAcc) / 2
        print(f">>> VALIDATION SCORE FOR EPOCH {epoch}: {np.round(val_score, 4)}, loss: {val_loss}")
        val_epochs.append((val_mIoU, val_allAcc, val_score))
        print(">>> COMPUTING TEST ERROR <<<")
        test_loss, test_mIoU, test_mAcc, test_allAcc = validate(model, data_list=test_list)
        test_score = (test_mIoU + test_allAcc) / 2
        print(f">>> TEST SCORE FOR EPOCH {epoch}: {np.round(test_score, 4)}, loss: {test_loss}")
        test_epochs.append((test_mIoU, test_allAcc, test_score))
        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, f"psp_d:{dist_dim}_k{top_k}_n:{norm}_d:{decay}_e:{epoch}v2.pth")
        
    print(f"Validation Epochs: {val_epochs}")
    print(f"Test Epochs: {test_epochs}")
    return val_epochs, test_epochs

if __name__ == "__main__":
    dist_dim, top_k, norm, decay = "k", 5, "ln", 1e-6
    if len(sys.argv) > 1:
        print("Parsing arguments (data_dim)")
        dist_dim = sys.argv[1]
        # top_k = int(sys.argv[2])
        # norm = sys.argv[3]
        # decay = float(sys.argv[4])
    print(f"Running with arguments: dist:{dist_dim}, k:{top_k}, norm:{norm}, decay:{decay}")
    model_conv = PSPNetContext(pspnet_weights="exp/ade20k/pspnet50/model/train_epoch_100.pth", top_k=top_k, dist_dim=dist_dim, norm=norm).to("cuda")
    val_hist_conv = main(model_conv, dist_dim, top_k, decay)
    print(val_hist_conv)  # print val results again
