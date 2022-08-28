import argparse
import glob
import logging
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

import acosp.inject
from acosp.pruner import SoftTopKPruner
from util import config
from util.util import colorize

cv2.ocl.setUseOpenCL(False)


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation")
    parser.add_argument(
        "--config",
        type=str,
        default="config/ade20k/ade20k_pspnet50.yaml",
        help="config file",
    )
    parser.add_argument("--image-dir", type=str, help="input images")
    parser.add_argument(
        "opts",
        help="see config/ade20k/ade20k_pspnet50.yaml for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg.image_dir = args.image_dir
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def check(args):
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    if args.arch == "psp":
        assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    elif args.arch == "segnet":
        pass
    else:
        raise Exception(f"architecture {args.arch} not supported yet")


def main():
    global args, logger
    args = get_parser()
    check(args)
    logger = get_logger()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.test_gpu)
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    predict_directory(
        args.arch,
        args.colors_path,
        args.layers,
        args.classes,
        args.zoom_factor,
        args.model_path,
        logger,
        args.base_size,
        args.test_h,
        args.test_w,
        args.image_dir,
        args.sparse,
        args.scales,
        args.demo_save_path,
    )


def predict_directory(
    arch: str,
    colors_path,
    layers,
    classes,
    zoom_factor,
    model_path,
    logger,
    base_size,
    test_h,
    test_w,
    image_dir,
    sparse=False,
    scales=[1],
    save_path="./figures/demo",
):
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    colors = np.loadtxt(colors_path).astype("uint8")

    if arch == "psp":
        from model.pspnet import PSPNet

        model = PSPNet(
            layers=layers,
            classes=classes,
            zoom_factor=zoom_factor,
            pretrained=False,
        )
    elif arch == "segnet":
        from model.segnet import SegNet

        model = SegNet(classes=classes)
    logger.info(model)

    model = torch.nn.DataParallel(model).cpu()
    if sparse:
        pruner = SoftTopKPruner(ending_epoch=1, final_sparsity=0.1)
        pruner.configure_model(model)
        acosp.inject.soft_to_hard_k(model)

    cudnn.benchmark = True
    if os.path.isfile(model_path):
        logger.info("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])
        acosp.inject.hard_to_conv(model)
        logger.info("=> loaded checkpoint '{}'".format(model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(model_path))

    model.eval()
    model.cuda()

    rng = np.random.RandomState(42)
    all_files = sorted(glob.glob(image_dir))

    max_images = 40

    for fp in all_files if max_images > len(all_files) else rng.choice(all_files, max_images):
        test(
            model,
            fp,
            classes,
            mean,
            std,
            base_size,
            test_h,
            test_w,
            scales,
            colors,
            save_path,
        )


def net_process(model, image, mean, std=None, flip=True):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)
    input = input.unsqueeze(0).cuda()
    if flip:
        input = torch.cat([input, input.flip(3)], 0)
    with torch.no_grad():
        output = model(input)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode="bilinear", align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output


def scale_process(
    model,
    image,
    classes,
    crop_h,
    crop_w,
    h,
    w,
    mean,
    std=None,
    stride_rate=2 / 3,
):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(
            image,
            pad_h_half,
            pad_h - pad_h_half,
            pad_w_half,
            pad_w - pad_w_half,
            cv2.BORDER_CONSTANT,
            value=mean,
        )
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop, mean, std)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half : pad_h_half + ori_h, pad_w_half : pad_w_half + ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction


def test(
    model,
    image_path,
    classes,
    mean,
    std,
    base_size,
    crop_h,
    crop_w,
    scales,
    colors,
    save_path="./figure/demo",
):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
    h, w, _ = image.shape
    prediction = np.zeros((h, w, classes), dtype=float)
    for scale in scales:
        long_size = round(scale * base_size)
        new_h = long_size
        new_w = long_size
        if h > w:
            new_w = round(long_size / float(h) * w)
        else:
            new_h = round(long_size / float(w) * h)
        image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        prediction += scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
    prediction = scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
    prediction = np.argmax(prediction, axis=2)
    gray = np.uint8(prediction)
    color = colorize(gray, colors)
    image_name = image_path.split("/")[-1].split(".")[0]
    os.makedirs(save_path, exist_ok=True)
    gray_path = os.path.join(save_path, image_name + "_gray.png")
    color_path = os.path.join(save_path, image_name + "_color.png")
    cv2.imwrite(gray_path, gray)
    color.save(color_path)
    logger.info("=> Prediction saved in {}".format(color_path))


if __name__ == "__main__":
    main()
