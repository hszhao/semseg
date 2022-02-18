import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

import numpy as np
colors = np.loadtxt("/home/connor/Dev/semseg/data/ade20k/ade20k_colors.txt")
names = np.loadtxt("/home/connor/Dev/semseg/data/ade20k/ade20k_names.txt", dtype='str', usecols=0, delimiter="\n")


def get_sample(index):
    image = np.load(f"uper_samples/input{index}.npy")
    annot = mpimg.imread(f"uper_samples/annot{index}.png")
    pred = mpimg.imread(f"uper_samples/baseline{index}.png")
    corr = mpimg.imread(f"uper_samples/corrected{index}.png")

    dist_label = np.load("dist_targets.npz.npy")[index]
    dist_pred = np.load("dist_preds.npz.npy")[index]
    dist_alt_pred = np.load("dist_alt_preds.npz.npy")[index]

    return image, annot, pred, corr, dist_label, dist_pred, dist_alt_pred

def crop_images(images, crop_x, crop_y):
    return [image[crop_x+1:-crop_x, crop_y+1:-crop_y] for image in images]
    
def extract_dists(annot, pred, corr, k=5):
    annot, pred, corr = annot[:,:,:3], pred[:,:,:3], corr[:,:,:3]
    # predicted top_k
    pred_uniq, pred_counts = np.unique(pred.reshape(-1, 3), return_counts=True, axis=0)
    mask = [np.invert((pred_uniq == [255, 255, 255]).all(axis=1))]
    pred_uniq = pred_uniq[mask]
    pred_counts = pred_counts[mask]
    top_k_classes, top_k_counts = pred_uniq[np.argsort(-pred_counts)[:k]], pred_counts[np.argsort(-pred_counts)[:k]]
    top_k_counts = top_k_counts / top_k_counts.sum()

    # label top_k
    label_uniq, label_counts = np.unique(annot.reshape(-1, 3), return_counts=True, axis=0)
    label_k_counts = np.zeros_like(top_k_counts)
    for i in range(k):
        try:
            idx = np.argwhere((label_uniq == top_k_classes[i]).all(axis=1)).item()
            label_k_counts[i] = label_counts[idx]
        except:
            continue
    label_k_counts = label_k_counts / label_k_counts.sum()

    # adjusted top_k
    corr_uniq, corr_counts = np.unique(corr.reshape(-1, 3), return_counts=True, axis=0)
    corr_k_counts = np.zeros_like(top_k_counts)
    for i in range(k):
        try:
            idx = np.argwhere((corr_uniq == top_k_classes[i]).all(axis=1)).item()
            corr_k_counts[i] = corr_counts[idx]
        except:
            continue
    corr_k_counts = corr_k_counts / corr_k_counts.sum()

    return top_k_classes, top_k_counts, label_k_counts, corr_k_counts


def show_sample(image, annot, pred, corr, dist_label, dist_pred, dist_alt_pred, crop_x=128, crop_y=128):
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
    image, annot, pred, corr = crop_images([image, annot, pred, corr], crop_x=crop_x, crop_y=crop_y)
    axes[0][0].imshow(image)
    axes[0][0].axis("off")
    axes[0][0].title.set_text("Input Image")
    annot = np.asarray(annot * 255, dtype=np.uint8)
    axes[0][1].imshow(annot)
    axes[0][1].axis("off")
    axes[0][1].title.set_text("Ground Truth")
    pred = np.asarray(pred * 255, dtype=np.uint8)
    axes[0][2].imshow(pred)
    axes[0][2].axis("off")
    axes[0][2].title.set_text("Prediction")
    corr = np.asarray(corr * 255, dtype=np.uint8)
    axes[0][3].imshow(corr)
    axes[0][3].axis("off")
    axes[0][3].title.set_text("Distribution-Correction")

    top_k_classes, top_k_counts, label_k_counts, corr_k_counts = extract_dists(annot, pred, corr)
    bars = np.arange(len(top_k_counts))
    classes = []
    for i in range(len(bars)):
        try:
            c = names[np.argwhere((colors == top_k_classes[i]).all(axis=1))[0].item()]
            classes.append(c)
        except:
            continue
    axes[1][0].set_visible(False)
    axes[1][1].bar(x=bars, height=label_k_counts, color=top_k_classes/255)
    axes[1][1].set_xticks(bars)
    axes[1][1].set_xticklabels(classes)
    axes[1][2].bar(x=bars, height=top_k_counts, color=top_k_classes/255)
    axes[1][2].set_xticks(bars)
    axes[1][2].set_xticklabels(classes)
    axes[1][3].bar(x=bars, height=corr_k_counts, color=top_k_classes/255)
    axes[1][3].set_xticks(bars)
    axes[1][3].set_xticklabels(classes)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image, annot, pred, corr, dist_label, dist_pred, dist_alt_pred = get_sample(1708)
    show_sample(image, annot, pred, corr, dist_label, dist_pred, dist_alt_pred)


# top 5 gain - index | crop
# 558 | 128
# 535
# 108
# 65
# 1763

# top 5 worst - index | crop
# 557
# 1445
# 767
# 1027
# 1708