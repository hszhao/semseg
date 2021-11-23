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

from model.pspnet_c import PSPNetClassification

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
batch_size = 16
valid_list = "dataset/ade20k/list/validation.txt"
epochs=10
n_classes = 150

def mean_auc(classification_labels, classification_predictions):
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
        seg_target, class_target = target
        seg_target = seg_target.cuda(non_blocking=True)
        class_target = [ct.float().cuda(non_blocking=True) for ct in class_target]
        input = input.cuda(non_blocking=True)
        target = (seg_target, class_target)
        segmentation, classification, main_loss, aux_loss, classification_loss = model(input, target)
        main_loss, aux_loss, class_loss = torch.mean(main_loss), torch.mean(aux_loss), torch.mean(classification_loss)
        loss = class_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = input.size(0)

        intersection, union, target = intersectionAndUnionGPU(segmentation, seg_target, 150, 255)

        auc = mean_auc(class_target, classification)

        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter.update(aux_loss.item(), n)
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

        print('Epoch: [{}/{}][{}/{}] '
                'Auc: {}'.format(epoch+1, epochs, i + 1, len(train_loader), auc))

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
    aucs = []

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        seg_target, class_target = target
        seg_target = seg_target.cuda(non_blocking=True)
        class_target = [ct.float().cuda(non_blocking=True) for ct in class_target]
        input = input.cuda(non_blocking=True)
        # target = (seg_target, class_target)
        # seg_target = seg_target.cuda(non_blocking=True)
        # class_target = [ct.float().cuda(non_blocking=True) for ct in class_target]
        # input = input.cuda(non_blocking=True)
        # target = (seg_target, class_target)
        segmentation, classification = model(input, target)

        auc = mean_auc(class_target, classification)
        aucs.append(auc)

    print('Val result: auc ', np.mean(aucs))
    return np.mean(aucs)

def main():
    model = PSPNetClassification(pspnet_weights="exp/ade20k/pspnet50/model/train_epoch_100.pth").to("cuda")
    # params_list = [dict(params=model.classification_head.parameters(), lr=1e-2)]
    optimizer = torch.optim.Adam(params=model.classification_head.parameters(), lr=1e-3, weight_decay=1e-4)

    train_transform = transform.Compose([
    transform.RandScale([0.5, 2.0]),
    transform.RandRotate([-10, 10], padding=mean, ignore_label=255),
    transform.RandomGaussianBlur(),
    transform.RandomHorizontalFlip(),
    transform.Crop([473, 473], crop_type='rand', padding=mean, ignore_label=255),
    transform.ToTensor(),
    transform.Normalize(mean=mean, std=std)])
    train_data = dataset.SemData(split='train', data_root=data_root, data_list=train_list, transform=train_transform, classification_heads_x=False, classification_heads_y=True)
    val_results = []
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, sampler=None, drop_last=True)
    for epoch in range(0, epochs):
        epoch_log = epoch + 1
        train(train_loader, model, optimizer, epoch)
        val_auc = validate(model)
        val_results.append(val_auc)

    print(val_results)
    torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, "classification_v2.pth")

if __name__ == "__main__":
    main()