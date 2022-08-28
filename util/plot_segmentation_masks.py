import itertools
from collections import defaultdict
from glob import glob
from typing import Dict, List

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import cm


def get_segmentation_maps(prediction_folder: str, postfixes: List[str]):
    all_files = itertools.chain(*(sorted(glob(f"{prediction_folder}/**/*{postfix}")) for postfix in postfixes))
    # print(len(all_files))
    collate_dct = defaultdict(dict)

    for file in all_files:
        seg, fname = file.split("/")[-2:]
        for postfix in postfixes:
            fname = fname.replace(postfix, "")
        collate_dct[fname][seg] = file

    return collate_dct


def load_data(dct, keys):
    maps = dict()
    for k in keys:
        img = Image.open(dct[k])
        # img = cv2.imread(dct[k])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        maps[k] = np.asarray(img)

    return maps


def remap(img, class_map):
    new_img = img.copy()
    for new_c, old_c in class_map.items():
        new_img[img == old_c] = new_c
    return new_img


def plot_seg_maps(imgs: List[Dict[str, np.ndarray]], titles: Dict[str, str]):
    unique_classes = sorted(
        set(y for x in (tuple(np.unique(img)) for m in imgs for k, img in m.items() if k != "real") for y in x)
    )
    unique_classes = {i + 1: x for i, x in enumerate(unique_classes)}
    unique_classes[0] = 0

    num_classes = len(unique_classes)

    viridis = cm.get_cmap("plasma", num_classes)
    fig, axs = plt.subplots(nrows=len(imgs), ncols=len(titles), figsize=(12, 4.2))
    for idx, record in enumerate(imgs):
        for ax, (name, img) in zip(axs[idx, :], record.items()):
            if name != "real":
                old_img = img
                img += 1
                img[np.where(old_img == 256)] = 0
                img = remap(img, unique_classes)

            _f = ax.matshow(img, cmap=viridis)
            _f.set_clim(0, num_classes)
            ax.axis("off")
            if idx == 0:
                ax.set_title(titles[name])
            #    _f.set_clim(0, 10)
            # else:
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    return fig


def make_titles(model_key, model_title):
    post_fixes = [
        ("", "uncompressed"),
        ("_0.5", "ACoSP-2"),
        ("_0.75", "ACoSP-4"),
        ("_0.875", "ACoSP-8"),
        ("_0.9375", "ACoSP-16"),
    ]
    return {
        "real": "RGB",
        "ground": "Ground truth",
        **{f"{model_key}{pf_key}": f"{model_title}\n@ {pf_val}" for pf_key, pf_val in post_fixes},
    }


def main(
    prediction_folder: str,
    model_key: str = "pspnet50",
    model_title: str = "PSPNet-50",
    prediction_postfixes: List[str] = ("color.png",),
    choose_files: List[str] = None,
):
    titles = make_titles(model_key, model_title)
    if isinstance(prediction_postfixes, str):
        prediction_postfixes = prediction_postfixes.split(",")
    if isinstance(choose_files, str):
        choose_files = choose_files.split(",")
    dct = get_segmentation_maps(prediction_folder, prediction_postfixes)
    test_k = list(dct.keys())
    imgs = list()

    rng = np.random.RandomState(5)
    file_iter = (test_k[idx] for idx in rng.permutation(range(0, len(test_k))))

    if choose_files:
        file_iter = itertools.chain(choose_files, file_iter)

    for _k in file_iter:
        try:
            imgs.append(load_data(dct[_k], titles.keys()))
            print(f"choosing {_k}")
        except KeyError as e:
            print(f"Skipping {_k} because of missing images: {e}")

        if len(imgs) == 4:
            break
    else:
        raise ValueError("Failed to find enough images.")

    plot_seg_maps(imgs, titles)
    plt.savefig("./segmentation_degradation.png", dpi=1024)
    # plt.show()


if __name__ == "__main__":
    fire.Fire(main)
