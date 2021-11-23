import os
import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.init as initer


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step_learning_rate(base_lr, epoch, step_epoch, multiplier=0.1):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    return lr


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):
    """
    :param model: Pytorch Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param batchnorm: 'normal' or 'constant'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """
    for m in model.modules():
        if isinstance(m, (nn.modules.conv._ConvNd)):
            if conv == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, (nn.modules.batchnorm._BatchNorm)):
            if batchnorm == 'normal':
                initer.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                initer.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        initer.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        initer.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    initer.constant_(param, 0)


def group_weight(weight_group, module, lr):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

import cv2
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(1, os.path.abspath('..'))
from model.pspnet_c import PSPNetClassification
from model.pspnet import PSPNet

from classification_utils import extract_mask_classes

palette = np.loadtxt("/home/connor/Dev/semseg/data/ade20k/ade20k_colors.txt").astype('uint8')
label5 = cv2.imread("/home/connor/Dev/semseg/dataset/ade20k/annotations/training/ADE_train_00001049.png", 0)
pspnet_weights= "/home/connor/Dev/semseg/classification_v2_ep4.pth"

ratios = np.loadtxt("/home/connor/Dev/semseg/dataset/ade20k/objectInfo150.txt", delimiter="\t", dtype=str)[1:]


value_scale = 255
mean = [0.485, 0.456, 0.406]
# mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
# std = [item * value_scale for item in std]

filepath="/home/connor/Dev/semseg/dataset/ade20k/images/training/ADE_train_00001049.jpg"
classes = np.loadtxt("/home/connor/Dev/semseg/data/ade20k/ade20k_names.txt", dtype=str, delimiter="\n")
def pred_image(model, image):
    size = image.shape[1], image.shape[0]
    image = image / 255.0
    image = image - mean
    image = image / std
    ori_h, ori_w, _ = image.shape
    pad_h = max(473 - ori_h, 0)
    pad_w = max(473 - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, 
                                    pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, 
                                    cv2.BORDER_CONSTANT, value=mean)
    image = cv2.resize(image, (473, 473), interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    image = image.permute(0, 3, 1, 2).float().to("cuda")
    pred, classif = model(image)
    pre_softmax = pred.clone()
    pred_round = pred.clone().max(1)[1]
    pred_round = pred_round.to("cpu").numpy()[0]
    pred_round = pred_round.astype(np.uint8)
    pred_round = pred_round[pad_h_half:473-(pad_h_half+1), pad_w_half:473-(pad_w_half+1)]
    # pred = cv2.resize(pred, (size), interpolation=cv2.INTER_NEAREST)
    # pred = torch.from_numpy(pred).max(0)[1].numpy()
    return pred_round, classif, pre_softmax
    
import torch.nn.functional as F


def get_softmax_prediction_accuracy(feature_map, label, channel=12):
    sm = feature_map.max(0)[1]
    y_pred = np.where(sm.numpy() == channel, 1, 0)
    y_true = np.where(label == channel, 1, 0)
    total_err = np.sum(np.abs(y_pred - y_true))
    correct = np.sum(y_pred * y_true)
    total = np.sum(y_true)
    return correct / (total), total_err

def sample_class_ratios(classifications):
    classifications = classifications.numpy().reshape(150, -1)
    num_cells_masked_per_class = np.sum(classifications, axis=1)
    num_cells_marked_total = np.sum(classifications, axis=None)
    per_class_ratio = num_cells_masked_per_class / (num_cells_marked_total + 1e-7)
    return per_class_ratio

def sample_softmax_ratios(pre_softmax):
    features = pre_softmax.numpy().reshape(150, -1)
    num_cells_masked_per_class = np.sum(features, axis=1)
    num_cells_marked_total = np.sum(features, axis=None)
    per_class_ratio = num_cells_masked_per_class / (num_cells_marked_total + 1e-7)
    return per_class_ratio

def experiment(pre_softmax, classifications, alpha=0.5, label=None, n=5, tensor=True):
    # 1x1 scale
    pre_softmax = pre_softmax.clone()[0]
    class_ratios = ratios[:,1].astype(np.float32)
    class_ratios = torch.from_numpy(class_ratios).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    class_ratios = F.interpolate(class_ratios, (473, 473), mode="nearest")[0]
    ori_h, ori_w = label.shape
    pad_h = max(473 - ori_h, 0)
    pad_w = max(473 - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        label = cv2.copyMakeBorder(label, pad_h_half, 
                                    pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, 
                                    cv2.BORDER_CONSTANT, value=255)
    pre_softmax = pre_softmax.to("cpu")

    # tools: classification mask, class_ratios, pre_softmax
    # IDEA: ratio of class in prediction should be equal to that of class in pre_softmax
    classifications = torch.from_numpy(classifications)
    positive_softmax = (pre_softmax > 0)*1.0*pre_softmax
    positive_softmax_ratios = sample_softmax_ratios(positive_softmax)
    classification_ratios = sample_class_ratios(classifications)
    class_weights = classification_ratios / (positive_softmax_ratios+1e-7)
    class_weights = class_weights.reshape(-1, 1, 1)
    reweighted_pre_softmax = pre_softmax * class_weights
    new_softmax = (reweighted_pre_softmax*alpha) + pre_softmax
    return new_softmax
    
    # label = torch.from_numpy(label)
    # classification_mask = F.interpolate(classifications.unsqueeze(0), (473, 473), mode="nearest")[0]
    # pre_softmax = pre_softmax
    # class_ratios = class_ratios

    # # we now have Cx473x473 maps for pre_softmax and for our ratios
    # one = None
    # if tensor:
    #     one = classifications[0].cpu().detach().numpy()[0,:,0,0]
    # else:
    #     one = classifications[0][:,0,0]
    # top_n_ind = np.argpartition(one, -n)[-n:]
    # print(f"Top {n} values, classes, at 1x1:")
    # for i in top_n_ind:
    #     print(i, classes[i], one[i], ratios[i][1], torch.sum(pre_softmax[i,:,:]))
    # print(f"Top {n} values, classes, at 2x2:")
    # for x in range(2):
    #     for y in range(2):
    #         print(f"Region {x}, {y}:")
    #         if tensor:
    #             two = classifications[1].cpu().detach().numpy()[0,:,x,y]
    #         else:
    #             two = classifications[1][:,x, y]
    #         top_n_ind = np.argpartition(two, -n)[-n:]
    #         for i in top_n_ind:
    #             print(i,classes[i], two[i], ratios[i][1], torch.sum(pre_softmax[i,x,y]))
    # return

if __name__ == "__main__":
    pspnet = PSPNetClassification(layers=50, classes=150, zoom_factor=8, pspnet_weights=None).to("cuda")
    checkpoint = torch.load(pspnet_weights)['state_dict']
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    classification_labels = extract_mask_classes(label5)
    plt.imshow(image)
    plt.show()
    pspnet.load_state_dict(checkpoint)
    pspnet.eval()
    pred_seg, pred_class, pre_softmax = pred_image(pspnet, image)
    
    # i = colorize(pred_seg, palette)
    # plt.imshow(i)
    # plt.show()

    alphas = [1.0, 0.3, 0.2, 0.1, 0]
    fig, axes = plt.subplots(nrows=1, ncols=len(alphas)+1, figsize=(20, 12))
    real_classification = pred_class[-1].cpu().detach().numpy()[0]
    true_classification = classification_labels[-1]
    noisy_classification = (real_classification + true_classification) / 2
    axes[0].imshow(colorize(cv2.resize(label5, dsize=(473, 473), interpolation=cv2.INTER_NEAREST), palette))
    # axes[0].imshow(cv2.resize(image, dsize=(473, 473)))
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    # plt.show()
    for a in range(len(alphas)):
        new_pre_softmax_1 = experiment(pre_softmax, real_classification, label=label5, n=5, tensor=False, alpha=alphas[a])
        new_seg_1 = new_pre_softmax_1.max(0)[1].numpy()
        i = colorize(new_seg_1, palette)
        axes[a+1].imshow(i)
        axes[a+1].set_title(f"Classification Bias (a={alphas[a]})")
        axes[a+1].axis("off")
        if alphas[a] == 0:
            axes[a+1].set_title("Original Prediction (a=0)")
        
    plt.show()

    # new_pre_softmax_2 = experiment(pre_softmax, classification_labels, label=label5, n=5, tensor=False, alpha=1.0)
    # new_seg_2 = new_pre_softmax_2.max(0)[1].numpy()
    # i = colorize(new_seg_2, palette)
    # plt.imshow(i)
    # plt.show()

    # new_pre_softmax_3 = experiment(pre_softmax, classification_labels, label=label5, n=5, tensor=False, alpha=0.25)
    # new_seg3 = new_pre_softmax_3.max(0)[1].numpy()
    # i = colorize(new_seg3, palette)
    # plt.imshow(i)
    # plt.show()

    # new_pre_softmax_4 = experiment(pre_softmax, classification_labels, label=label5, n=5, tensor=False, alpha=0.1)
    # new_seg_4 = new_pre_softmax_4.max(0)[1].numpy()
    # i = colorize(new_seg_4, palette)
    # plt.imshow(i)
    # plt.show()

