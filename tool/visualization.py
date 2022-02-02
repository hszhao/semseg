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
from tool.test import multiscale_prediction

from model.pspnet_context import PSPNetContext
from model.pspnet_ms_context import PyramidContextNetwork
from model.encnet import EncNet
from model.fcn import FCN

import pandas as pd
import seaborn as sns

from sklearn.metrics import roc_auc_score

value_scale = 255
mean = [0.485, 0.456, 0.406]
mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
std = [item * value_scale for item in std]

data_root = "dataset/ade20k"
train_list = "dataset/ade20k/list/training.txt"
valid_list = "dataset/ade20k/list/validation.txt"
batch_size = 8
epochs = 100
n_classes = 150

palette = np.loadtxt("/home/connor/Dev/semseg/data/ade20k/ade20k_colors.txt").astype('uint8')
classes = pd.read_csv("/home/connor/Dev/semseg/dataset/ade20k/classes.csv")["Name"].values

def get_unnormalized_image(image):
    input_image_viz = image.squeeze().permute((1, 2, 0)).cpu()
    input_image_viz = input_image_viz * torch.tensor(std)
    input_image_viz = input_image_viz + torch.tensor(mean)
    input_image_viz = input_image_viz.cpu().numpy().astype(np.uint8)
    return input_image_viz

def dist_prediction_visualization(model):
    """
    TODO refactor
    """
    val_transform = transform.Compose([
        transform.Crop([473, 473], crop_type='center', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    val_data = dataset.SemData(split='val', data_root=data_root, data_list=valid_list, transform=val_transform, context_x=False, context_y=True)
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

    dist_pred_baseline = []
    dist_pred_trained = []
    dist_targets = []
    for i, (input, target) in enumerate(val_loader):
        seg_target, dist_target = target
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        seg_target = seg_target.cuda(non_blocking=True)
        dist_target = dist_target[0]
        dist_targets.append(dist_target)
        dist_target = dist_target.cuda(non_blocking=True)
        predicted_distribution = predicted_distribution[0]

        # colors = np.loadtxt("/home/connor/Dev/semseg/data/ade20k/ade20k_colors.txt").astype('uint8')
        # fig, axes = plt.subplots(2, 5, figsize=(12, 14))
        # viz_image = get_unnormalized_image(input)
        # gt_seg = colorize(seg_target.squeeze().cpu().numpy().astype(np.uint8), colors)
        # prediction_reg = colorize(output.max(1)[1].squeeze().cpu().numpy().astype(np.uint8), colors)
        # prediction_alt = colorize(output_alt.max(1)[1].squeeze().cpu().numpy().astype(np.uint8), colors)
        # prediction_alt_gt = colorize(output_alt_gt.max(1)[1].squeeze().cpu().numpy().astype(np.uint8), colors)
        # axes[0][0].imshow(viz_image)
        # axes[0][0].set_title("Input Image")
        # axes[0][1].imshow(gt_seg)
        # axes[0][1].set_title("Ground Truth Segmentation")
        # axes[0][2].imshow(prediction_reg)
        # axes[0][2].set_title("Baseline PSPNet Segmentation")
        # axes[0][3].imshow(prediction_alt)
        # axes[0][3].set_title("PSPNet Segmentation + DistFix")
        # axes[0][4].imshow(prediction_alt_gt)
        # axes[0][4].set_title("PSPNet Segmentation + DistFix w/GT Labels")

        # dist_pred_trained.append(predicted_distribution)
        # predictions = set(np.unique(output.max(1)[1].squeeze().cpu().numpy()))
        # p2 = set(np.unique(output_alt.max(1)[1].squeeze().cpu().numpy())) 
        # p3 = set(np.unique(output_alt_gt.max(1)[1].squeeze().cpu().numpy()))
        # predictions.update(p2)
        # predictions.update(p3)

        # N_CLASS length vectors 
        dist_label = dist_target.squeeze().cpu().numpy()
        dist_baseline = extract_mask_distributions(output.max(1)[1].squeeze().cpu(), head_sizes=[1])[0].squeeze()
        dist_fix = extract_mask_distributions(output_alt.max(1)[1].squeeze().cpu(), head_sizes=[1])[0].squeeze()
        dist_fix_gt = extract_mask_distributions(output_alt_gt.max(1)[1].squeeze().cpu(), head_sizes=[1])[0].squeeze()

        # ERROR
        baseline_err = np.sum(np.abs(dist_label - dist_baseline))
        fix_err = np.sum(np.abs(dist_label - dist_fix))
        fix_gt_err = np.sum(np.abs(dist_label - dist_fix_gt))

        # predictions = list(predictions)
        # sns.barplot(x=[n for n in predictions], y=np.take(dist_label, [predictions], 0)[0], ax=axes[1][1])
        # sns.barplot(x=[n for n in predictions], y=np.take(dist_baseline, [predictions], 0)[0], ax=axes[1][2])
        # axes[1][2].set_xlabel(f"Distance: {np.round(baseline_err, decimals=2)}")
        # sns.barplot(x=[n for n in predictions], y=np.take(dist_fix, [predictions], 0)[0], ax=axes[1][3])
        # axes[1][3].set_xlabel(f"Distance: {np.round(fix_err, decimals=2)}")
        # sns.barplot(x=[n for n in predictions], y=np.take(dist_fix_gt, [predictions], 0)[0], ax=axes[1][4])
        # axes[1][4].set_xlabel(f"Distance: {np.round(fix_gt_err, decimals=2)}")

        # plt.savefig(f"samples/sample{i}.png")
        # plt.clf()

        # dist_pred_baseline.append(dist_baseline)

        n = input.size(0)

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, seg_target, 150, 255)
        
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        ious.append(intersection / (union + 1e-10))
        accs.append(intersection / (target + 1e-10))

        output_alt_gt = output_alt_gt.max(1)[1]
        intersection2, union2, target2 = intersectionAndUnionGPU(output_alt_gt, seg_target, 150, 255)
        
        intersection2, union2, target2 = intersection2.cpu().numpy(), union2.cpu().numpy(), target2.cpu().numpy()
        intersection_meter2.update(intersection2), union_meter2.update(union2), target_meter2.update(target2)

        ious2.append(intersection2 / (union2 + 1e-10))
        accs2.append(intersection2 / (target2 + 1e-10))

        batch_time.update(time.time() - end)
        end = time.time()
        # if i > 30:
        #     break

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

def extract_ppm_features(x, ppm):
    x0 = ppm.features[0][1](x)
    x0 = ppm.features[0][2](x0)
    x0 = ppm.features[0][3](x0)

    x1 = ppm.features[1][1](x)
    x1 = ppm.features[1][2](x1)
    x1 = ppm.features[1][3](x1)

    x2 = ppm.features[2][1](x)
    x2 = ppm.features[2][2](x2)
    x2 = ppm.features[2][3](x2)

    x3 = ppm.features[3][1](x)
    x3 = ppm.features[3][2](x3)
    x3 = ppm.features[3][3](x3)
    return x0, x1, x2, x3   

def classification_heatmap(xs, input, model, multi_scale = True):
    """
    pyramid classification --> single heatmap
    """
    features = []
    for x in xs:
        x = nn.ReLU()(model.pyramid.bn(model.pyramid.conv1(x)))
        x = nn.ReLU()(model.pyramid.class_head(x))  # no activation maybe uncertain?
        x = F.interpolate(x, size=input.size()[2:], mode="bilinear", align_corners=True)
        features.append(x)
    if not multi_scale:
        return features[0].squeeze().cpu().detach().numpy()
    else:
        return torch.mean(torch.cat(features, dim=0), dim=0).squeeze().cpu().detach().numpy()

def cam(model, input, seg_target):
    """
    standard CAM, assumes classification layer immediately follows GAP of final features
    """
    output, _ = model.forward(input)

    dist = extract_mask_distributions(seg_target.squeeze().cpu())
    seg = output.max(1)[1].squeeze().detach().cpu().numpy()
    # forward pass up until ppm
    x = model.pspnet.pspnet.layer0(input)
    x = model.pspnet.pspnet.layer1(x)
    x = model.pspnet.pspnet.layer2(x)
    x = model.pspnet.pspnet.layer3(x)
    x = model.pspnet.pspnet.layer4(x)
    x = model.pspnet.pspnet.ppm(x)
    # extract final feature map
    for i in range(len(model.pspnet.pspnet.cls)-1):
        x = model.pspnet.pspnet.cls[i](x)
    # apply classification head and ReLU
    x = model.pspnet.prediction.classification_head(x)
    x = F.interpolate(x, size=input.size()[2:], mode="bilinear", align_corners=True)
    x = nn.Softmax(dim=1)(x)
    x = x.squeeze().cpu().detach().numpy()
    num_cam_rows = int(np.ceil(len(np.unique(seg)) / 3))
    fig, axes = plt.subplots(1 + num_cam_rows, 3, figsize=(12, 10))
    axes[0][0].imshow(get_unnormalized_image(input))
    axes[0][1].imshow(colorize(seg_target.squeeze().cpu().numpy(), palette))
    axes[0][2].imshow(colorize(seg, palette))
    cam_total = x.sum()
    seg_total = 473**2
    row, col = 1, 0
    classes_pred = np.unique(seg)
    classes_target = np.unique(seg_target.cpu().numpy())
    classes_target = classes_target[:-1] if 255 in classes_target else classes_target
    classes_viz = np.unique(np.concatenate([classes_pred, classes_target]))
    for cam_class in classes_viz:
        cam_class_map = x[cam_class]
        axes[row][col].imshow(cam_class_map)
        dist_prop = np.round((dist[0][cam_class][0][0] * 100), 3)
        cam_prop = np.round(((cam_class_map.sum()/cam_total) * 100), 3)
        seg_prop = np.round(((np.where(seg == cam_class, 1, 0).sum()/seg_total) * 100), 3)
        axes[row][col].set_xlabel(f"{classes[cam_class].split(';')[0]}: true:{dist_prop}%, cam: {cam_prop}%, seg: {seg_prop}% ")
        col += 1
        if col == 3:
            col = 0
            row += 1
    plt.tight_layout()
    plt.show()
    