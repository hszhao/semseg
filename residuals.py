import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# basic test seeing if class variance is correlated with segmentation performance

if __name__ == "__main__":
    ious = np.loadtxt("ious_encnet.gz")
    residuals = np.loadtxt("classification_residuals.gz")
    residuals = np.where(residuals == 0, np.nan, residuals)
    ious = np.where(ious == 0, np.nan, ious)
    df = pd.DataFrame(data=ious, columns=[f"iou_{i}" for i in range(150)])
    df[[f"resid_{i}" for i in range(150)]] = residuals
    corr = df.corr()
    corr_iou = corr[[f"iou_{i}" for i in range(150)]]
    corr_iou = corr_iou.iloc[150::]
    corr_iou = corr_iou.fillna(0)
    sns.heatmap(corr_iou)
    plt.show()