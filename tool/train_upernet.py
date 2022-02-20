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

from model.upernet import UPerNet

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
batch_size = 8
epochs = 25
n_classes = 150

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
        input = input.cuda(non_blocking=True)
        
        seg_target, context_target = target
        seg_target = seg_target.cuda(non_blocking=True)
        context_target = [ct.float().cuda(non_blocking=True) for ct in context_target]
        context_target = context_target[0] # only care single scale

        segmentation, loss = model(input, y=seg_target, context=context_target)
        loss = torch.mean(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = input.size(0)

        segmentation = segmentation.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(segmentation, seg_target, 150, 255)

        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(5e-3, current_iter, max_iter, power=0.9)
        print(f"Reducing LR to {current_lr}")
        for index in range(0, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr
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
        transform.Crop([512, 512], crop_type='center', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    val_data = dataset.SemData(split='val', data_root=data_root, data_list=data_list, transform=val_transform, context_x=False, context_y=True, context_type="classification")
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, sampler=None)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    loss_meter = AverageMeter()

    model.eval()
    end = time.time()
    ious = []
    accs = []

    for i, (input, target) in enumerate(val_loader):
        # seg_target = target
        seg_target, context_target = target
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        seg_target = seg_target.cuda(non_blocking=True)
        context_target = [ct.float().cuda(non_blocking=True) for ct in context_target]
        context_target = context_target[0]  # only look at global scale for now

        output, loss = model(input, y=seg_target, context=context_target)

        n = input.size(0)
        loss_meter.update(loss.item(), n)

        # baseline model
        output = output.max(1)[1]
        
        intersection, union, target = intersectionAndUnionGPU(output, seg_target, 150, 255)
        
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        ious.append(intersection / (union + 1e-10))
        accs.append(intersection / (target + 1e-10))

        batch_time.update(time.time() - end)
        end = time.time()
        print(f"{i+1}/{len(val_loader)}")

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    print('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))

    return loss_meter.avg, np.round(mIoU, 4), np.round(mAcc, 4), np.round(allAcc, 4)

def main(model, decay=1e-4, num_layers=1):
    # define optimizer
    learning_rate = 5e-3
    modules_new = [model.film_head.layer1, model.film_head.norm1, model.film_head.layer2]
    params_list = []
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=learning_rate))
    optimizer = torch.optim.Adam(params_list, lr=learning_rate, weight_decay=decay)
    
    train_transform = transform.Compose([
        transform.RandScale([0.5, 2.0]),
        transform.RandRotate([-10, 10], padding=mean, ignore_label=255),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([512, 512], crop_type='rand', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

    train_data = dataset.SemData(split='train', data_root=data_root, data_list=train_list, transform=train_transform, context_x=False, context_y=True, context_type="classification")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, sampler=None, drop_last=True)

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

        save_path = f"film_class_{epoch}_f{num_layers}v2.pth"
        print(f"Saving to {save_path}")
        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_path)
        
    print(f"Validation Epochs: {val_epochs}")
    print(f"Test Epochs: {test_epochs}")
    return val_epochs, test_epochs

classification_weights = "upernet_swin_classification_9_v2.pth"

if __name__ == "__main__":
    film_layers = 1 if len(sys.argv) < 2 else int(sys.argv[1])
    print(f"Training with {film_layers} new layers")
    model_conv = UPerNet(backbone="swin", learn_context=True, init_weights=classification_weights, context_layers=1, film=True, film_layers=2).to("cuda")
    val_hist_conv = main(model_conv, num_layers=film_layers)
    print(val_hist_conv)  # print val results again
