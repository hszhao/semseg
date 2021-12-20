import argparse
import os
from pathlib import Path

import numpy as np
from cv2 import cv2
from torchvision.datasets import Cityscapes

class_mapping = {cc.id: cc.train_id for cc in Cityscapes.classes}
inv_class_mapping = {cc.train_id: cc.id for cc in Cityscapes.classes}


def apply_to_mask(img: np.ndarray, mapping=class_mapping):
    new_img = img.copy()
    for k, v in mapping.items():
        new_img[img == k] = v
    return new_img


def convert_folder(input_dir, output_dir, mapping=class_mapping, img_postfix="labelIds.png"):
    fps = list(Path(input_dir).rglob(f"**/*{img_postfix}"))

    img_gen = (cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE) for fp in fps)
    converted_img_gen = (apply_to_mask(img, mapping) for img in img_gen)

    out_fps = (os.path.join(output_dir, str(fp.relative_to(input_dir))) for fp in fps)

    for fp, img in zip(out_fps, converted_img_gen):
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        cv2.imwrite(fp, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert cityscapes classes. Especially useful for converting training IDs to submission IDs."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--to_19", type=str, default="true")
    parser.add_argument("--img_postfix", type=str, default="labelIds.png")
    args = parser.parse_args()
    assert args.to_19 in {"true", "false"}

    convert_folder(
        args.input_dir,
        args.output_dir,
        mapping=(class_mapping if args.to_19 == "true" else inv_class_mapping),
        img_postfix=args.img_postfix,
    )
