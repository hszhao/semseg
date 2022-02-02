import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import cross_entropy
# visualize distributions of classes in gt and prediction

info = pd.read_csv("dataset/ade20k/objectInfo150.txt", sep="\t")
classes = info["Name"].values
classes_short = [c.split(",")[0] for c in classes]

dist = info["Ratio"].values

SHOW_PLOTS = True

def barplot(y_data, title=""):
    if not SHOW_PLOTS:
        return
    plt.figure(figsize=(9, 5), dpi=120)
    x_ticks = [i for i in range(len(classes))]
    sns.barplot(x=x_ticks, y=y_data)
    plt.xticks(x_ticks, labels=classes_short, rotation="vertical")
    plt.tight_layout()
    plt.title(title, fontsize=12)
    plt.show()

barplot(y_data=dist, title="Dataset Distribution (GT)")

# GT values for validation set
dist_gt = np.loadtxt("dist_gt_v3.gz")
# BATCHES, N_SAMPLES_BATCH = dist_gt.shape[0], dist_gt.shape[1]
# dist_gt = np.reshape(dist_gt, (BATCHES, N_SAMPLES_BATCH, 150))
# dist_gt_batch = dist_gt.mean(axis=1)
# dist_gt = np.reshape(dist_gt, (-1, 150))
dist_gt_global = np.sum(dist_gt, axis=0)
dist_gt_global = dist_gt_global / np.sum(dist_gt_global)

# barplot(y_data=dist_gt_global, title="Validation Set Distribution (GT)")

# baseline model
dist_pred_baseline = np.loadtxt("dist_pred_baseline_v3.gz")
# dist_pred_baseline = np.reshape(dist_pred_baseline, (BATCHES, N_SAMPLES_BATCH, 150))
# dist_pred_baseline_batch = dist_pred_baseline.mean(axis=1)
# dist_pred_baseline = np.reshape(dist_pred_baseline, (-1, 150))
dist_pred_baseline_global = np.sum(dist_pred_baseline, axis=0)
dist_pred_baseline_global = dist_pred_baseline_global / np.sum(dist_pred_baseline_global)

# barplot(y_data=dist_pred_baseline_global, title="Prediction Distribution (Baseline)")

# model using distribution head
dist_pred_trained = np.loadtxt("dist_pred_trained_v3.gz")
dist_pred_trained = np.loadtxt("dist_gt_corr.gz")
# dist_pred_trained = np.reshape(dist_pred_trained, (BATCHES, N_SAMPLES_BATCH, 150))
# dist_pred_trained_batch = dist_pred_trained.mean(axis=1)
# dist_pred_trained = np.reshape(dist_pred_trained, (-1, 150))
dist_pred_trained_global = np.sum(dist_pred_trained, axis=0)
dist_pred_trained_global = dist_pred_trained_global / np.sum(dist_pred_trained_global)

# barplot(y_data=dist_pred_trained_global, title="Prediction Distribution (Refit)")

# WASSERSTEIN DISTANCE
wasserstein_baseline = np.sum(np.abs(dist_gt - dist_pred_baseline)) / dist_gt.shape[0]
wasserstein_trained = np.sum(np.abs(dist_gt - dist_pred_trained)) / dist_pred_trained.shape[0]

print("BASELINE WASSERSTEIN:", wasserstein_baseline)
print("TRAINED WASSERSTEIN:", wasserstein_trained)

# % ERROR
dist_error_baseline = np.abs(dist_gt - dist_pred_baseline)
dist_error_trained = np.abs(dist_gt - dist_pred_trained)

cross_entropy_baseline = -dist_gt * np.log(dist_pred_baseline)
cross_entropy_trained = -dist_gt * np.log(dist_pred_trained)

# % ERROR IMP
dist_error_imp = (dist_error_baseline - dist_error_trained)
dist_avg_imp = np.mean(dist_error_imp, axis=0)

ce_avg_imp = np.mean((cross_entropy_baseline - cross_entropy_trained), axis=0)

barplot(y_data=ce_avg_imp, title="Mean Per-Class CE Improvement With DistFix (GT)")

# IOU SCORES
ious_baseline = np.loadtxt("baseline_ious_v3.gz")
# ious_trained = np.loadtxt("retrained_ious_v3.gz")
ious_trained = np.loadtxt("ious_corr.gz")

# IOU IMP
iou_imp = (ious_trained - ious_baseline)
avg_imp = np.mean(iou_imp, axis=0)

barplot(y_data=avg_imp, title="Mean Per-Class IOU Improvement With DistFix (GT)")

baseline_iou_err = np.hstack([dist_error_baseline, ious_baseline])
df_baseline = pd.DataFrame(baseline_iou_err, columns=[f"EMD{i}" for i in range(150)] + [f"IoU{i}" for i in range(150)])

corr_baseline = df_baseline.corr()
corr_baseline = corr_baseline.iloc[0:150, 150::]

plt.figure(figsize=(16, 14))
plt.tight_layout()
sns.heatmap(np.abs(corr_baseline), cmap="Blues")
plt.title("Correlation of IoU and Wasserstein - Baseline PSPNet")
plt.show()

diag = np.diagonal(np.abs(corr_baseline))
print(f"Mean Correlation Between IoU and Distribution % Error: {np.mean(diag)}")


baseline_iou_err = np.hstack([(cross_entropy_baseline - cross_entropy_trained), iou_imp])
df_baseline = pd.DataFrame(baseline_iou_err, columns=[f"dCE{i}" for i in range(150)] + [f"dIoU{i}" for i in range(150)])

corr_baseline = df_baseline.corr()
corr_baseline = corr_baseline.iloc[0:150, 150::]

plt.figure(figsize=(16, 14))
plt.tight_layout()
sns.heatmap(np.abs(corr_baseline), cmap="Blues")
plt.title("Correlation of IoU and Cross Entropy - Baseline PSPNet")
plt.show()

diag = np.diagonal(np.abs(corr_baseline))
print(f"Mean Correlation Between IoU and Distribution % Error: {np.mean(diag)}")

# print(f"Median Correlation Between IoU and Distribution % Error: {np.median(diag)}")

# plt.plot([i for i in range(150)], diag)
# plt.plot([i for i in range(150)], dist_gt_global)
# plt.show()
