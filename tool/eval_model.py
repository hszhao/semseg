import os
import random
import time
from turtle import color
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

from model.pspnet_context import PSPNetContext
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

palette = np.loadtxt("/home/connor/Dev/semseg/data/ade20k/ade20k_colors.txt").astype('uint8')

data_root = "dataset/ade20k"
# parition 10% of training set for hyperparameter tuning, to report real results on validation set.
train_list = "dataset/ade20k/list/training_alt.txt"
valid_list = "dataset/ade20k/list/validation_alt.txt"
test_list = "dataset/ade20k/list/validation.txt"
batch_size = 1
epochs = 1
n_classes = 150

def validate(model, data_list=valid_list):
    val_transform = transform.Compose([
        transform.Crop([512, 512], crop_type='center', padding=mean, ignore_label=255),
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

    model.eval()
    end = time.time()
    ious = []
    accs = []
    ious2 = []
    accs2 = []

    inputs = []
    targets = []
    preds = []
    alt_preds = []

    dist_targets = []
    dist_preds = []
    alt_dist_preds = []

    for i, (input, target) in enumerate(val_loader):
        seg_target, context_target = target
        data_time.update(time.time() - end)
        
        viz_input = np.asarray(mean + (input.squeeze().numpy().swapaxes(0, 2).swapaxes(0, 1) * std), dtype=np.uint8)
        input = input.cuda(non_blocking=True)

        seg_target = seg_target.cuda(non_blocking=True)
        context_target = [ct.float().cuda(non_blocking=True) for ct in context_target]
        context_target = context_target[0]  # only look at global scale for now

        output, _ = model(input, distribution=None)
        output_alt, k_label, k_softmax, k_softmax_alt = model(input, distribution=context_target)

        output = output.max(1)[1]
        output_alt = output_alt.max(1)[1]
        
        intersection, union, target = intersectionAndUnionGPU(output, seg_target, 150, 255)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
        ious.append(intersection / (union + 1e-10))
        accs.append(intersection / (target + 1e-10))

        intersection2, union2, target2 = intersectionAndUnionGPU(output_alt, seg_target, 150, 255)
        intersection2, union2, target2 = intersection2.cpu().numpy(), union2.cpu().numpy(), target2.cpu().numpy()
        intersection_meter2.update(intersection2), union_meter2.update(union2), target_meter2.update(target2)
        ious2.append(intersection2 / (union2 + 1e-10))
        accs2.append(intersection2 / (target2 + 1e-10))

        gt_color = colorize(seg_target.squeeze().cpu().numpy(), palette)
        pred1_color = colorize(output.squeeze().cpu().numpy(), palette)
        pred2_color = colorize(output_alt.squeeze().cpu().numpy(), palette)

        inputs.append(viz_input)
        targets.append(gt_color)
        preds.append(pred1_color)
        alt_preds.append(pred2_color)

        dist_targets.append(k_label.squeeze().cpu().numpy())
        dist_preds.append(k_softmax.squeeze().cpu().detach().numpy())
        alt_dist_preds.append(k_softmax_alt.squeeze().cpu().detach().numpy())
        
        batch_time.update(time.time() - end)
        end = time.time()
        print(f"{i+1}/{len(val_loader)}")

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    print('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))

    ious = np.asarray(ious).mean(axis=1)
    ious2 = np.asarray(ious2).mean(axis=1)
    improvement = ious2 - ious


    return np.round(mIoU, 4), np.round(mAcc, 4), np.round(allAcc, 4)

def main(model):
    test_epochs = [ ]
    for epoch in range(0, epochs):
        print(">>> COMPUTING TEST ERROR <<<")
        test_mIoU, test_mAcc, test_allAcc = validate(model, data_list=test_list)
        test_score = (test_mIoU + test_allAcc) / 2
        print(f">>> TEST SCORE FOR EPOCH {epoch}: {np.round(test_score, 4)}")
        test_epochs.append((test_mIoU, test_allAcc, test_score))

    print(f"Test Epochs: {test_epochs}")
    return test_epochs

if __name__ == "__main__":
    model = UPerNet(backbone="swin").to("cuda")
    result = main(model)
    print(result)
