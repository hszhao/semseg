import os
import random
import time
import cv2
import numpy as np
import logging
import argparse
from pandas.core.indexes import base

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist

from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(1, os.path.abspath('..'))

from util import dataset, transform, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, find_free_port

from model.pspnet_c import PSPNetClassification
from model.resnet_dist import ResNetDist

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
batch_size = 8
valid_list = "dataset/ade20k/list/validation.txt"
epochs=5
n_classes = 150

def mean_auc(classification_labels, classification_predictions):
    class_aucs = [0] * n_classes
    class_counts = [0] * n_classes
    # loop over each classification head
    for scale_idx in range (1):
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

def distribution_distance(distribution_labels, distribution_predictions):
    distances = []
    for scale in range(len(distribution_labels)):
        y_true, y_pred = distribution_labels[scale], distribution_predictions[scale]
        y_true, y_pred = y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
        distance = np.sum(np.abs(y_true - y_pred))
        distances.append(distance)
    return distances
 
def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    # step lr decay
    base_lr = 1e-3
    decay = 0.5
    epoch_lr = base_lr * (decay**epoch)
    print("reducing learning rate to", epoch_lr)
    optimizer.param_groups[0]["lr"] = epoch_lr

    model.train()
    end = time.time()
    max_iter = epochs * len(train_loader)
    for i, (input, target) in tqdm(enumerate(train_loader)):
        data_time.update(time.time() - end)
        seg_target, class_target = target
        seg_target = seg_target.cuda(non_blocking=True)
        class_target = [ct.float().cuda(non_blocking=True) for ct in class_target]
        input = input.cuda(non_blocking=True)
        target = seg_target, class_target
        segmentation, distributions, main_loss, aux_loss, distributions_loss = model(input, target)
        main_loss, aux_loss, distributions_loss = torch.mean(main_loss), torch.mean(aux_loss), torch.mean(distributions_loss)
        loss = distributions_loss

        # distributions, loss = model(input, target)
        # loss = torch.mean(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = input.size(0)
        dist_metric = distribution_distance(class_target, distributions)

        print(f'Batch {i+1}/{len(train_loader)}, Loss: {loss}, Distance {dist_metric[0]/(batch_size)}')

def validate(model):
    val_transform = transform.Compose([
        transform.Crop([473, 473], crop_type='center', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    val_data = dataset.SemData(split='val', data_root=data_root, data_list=valid_list, transform=val_transform, classification_heads_x=False, classification_heads_y=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, sampler=None)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    metrics = []

    model.eval()
    end = time.time()
    for i, (input, target) in tqdm(enumerate(val_loader)):
        data_time.update(time.time() - end)
        seg_target, class_targets = target
        seg_target = seg_target.cuda(non_blocking=True)
        class_targets = [ct.float().cuda(non_blocking=True) for ct in class_targets]
        input = input.cuda(non_blocking=True)
        # target = (seg_target, class_target)
        # seg_target = seg_target.cuda(non_blocking=True)
        # class_target = [ct.float().cuda(non_blocking=True) for ct in class_target]
        # input = input.cuda(non_blocking=True)
        # target = (seg_target, class_target)
        segmentation, distributions = model(input, class_targets)
        # distributions = model(input, seg_target)

        dist_metric = distribution_distance(class_targets, distributions)

        metrics.append(dist_metric)

    print('Val result: ', np.mean(metrics) / batch_size)
    return np.mean(metrics) / batch_size

def main():
    model = PSPNetClassification(pspnet_weights="exp/ade20k/pspnet50/model/train_epoch_100.pth").to("cuda")
    # checkpoint = torch.load("/home/connor/Dev/semseg/distributions0451.pth")['state_dict']
    # model.load_state_dict(checkpoint)
    # model = ResNetDist(size=2).to("cuda")
    # modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
    # modules_new = [model.cls]
    # params_list = []
    base_lr = 1e-3
    # for module in modules_ori:
    #     params_list.append(dict(params=module.parameters(), lr=base_lr))
    # for module in modules_new:
    #     params_list.append(dict(params=module.parameters(), lr=base_lr*10))
    # params_list = [dict(params=model.classification_head.parameters(), lr=1e-2)]
    params_list = [dict(params=c[1].parameters(), lr=base_lr) for c in model.pred.named_children() if c[0] != 'cls']

    optimizer = torch.optim.Adam(params_list, lr=base_lr,weight_decay=1e-4)

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
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, sampler=None, drop_last=True)
    for epoch in range(0, epochs):
        train(train_loader, model, optimizer, epoch)
        val_metric = validate(model)
        val_results.append(val_metric)
        
       
        # val_auc = validate(model)
        # val_results.append(val_auc)
        
    print(val_results)
    torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, "distributions.pth")

if __name__ == "__main__":
    main()