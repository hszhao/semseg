import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

scene = pd.read_csv('/home/connor/Dev/semseg/dataset/ade20k/sceneCategories.txt', sep=" ", header=None)
scene[1] = scene[1].str.rstrip()
unique_scenes = scene[1].unique()

classes = pd.read_csv("/home/connor/Dev/semseg/dataset/ade20k/classes.csv")["Name"].values

scene_dict = { unique_scenes[i]: i for i in range(len(unique_scenes))}
scene_counts = { i:0 for i in range(len(unique_scenes))}
scene_totals = { i:0 for i in range(len(unique_scenes))}

if __name__ == "__main__":
    base_iou = np.loadtxt("base_ious.gz")
    base_miou = np.mean(base_iou, axis=1)
    base_acc = np.loadtxt("base_accs.gz")
    base_macc = np.mean(base_acc, axis=1)

    new_iou = np.loadtxt("logit_ious.gz")
    new_miou = np.mean(new_iou, axis=1)
    new_acc = np.loadtxt("logit_accs.gz")
    new_macc = np.mean(new_acc, axis=1)

    count = 0
    class_counts = [0] * 150
    class_totals = [0] * 150

    max_iou_imprv = 0
    max_miou_imprv = 0
    for i in range(len(base_iou)):
        for c in range(150):
            if new_iou[i][c] > base_iou[i][c]:
                if new_iou[i][c] - base_iou[i][c] > max_iou_imprv:
                    max_iou_imprv = new_iou[i][c] - base_iou[i][c]
                class_counts[c] += 1
            if base_iou[i][c] > 0 or new_iou[i][c] > 0:
                class_totals[c] += 1

    class_imp = [class_counts[i] / class_totals[i] for i in range(len(class_totals)) ]
    for c in range(len(class_imp)):
        if class_imp[c] >= 0.5:
            print(f"{class_imp[c]*100}% of class {classes[c]} improved")

    for i in range(len(base_iou)):
        if new_miou[i] >= base_miou[i]:
            if new_miou[i] - base_miou[i] > max_miou_imprv:
                    max_miou_imprv = new_miou[i] - base_miou[i]
            scene_counts[scene_dict[scene.iloc[i][1]]] += 1
        scene_totals[scene_dict[scene.iloc[i][1]]] += 1

    for i in range(len(unique_scenes)):
        if scene_totals[i] < 5:
            continue
        percent_improved = scene_counts[i] / scene_totals[i]
        print(f"({scene_counts[i]}/{scene_totals[i]}), {percent_improved*100}% of {unique_scenes[i]} improved")

    plt.figure()
    plt.scatter([x for x in range(len(base_iou))], np.mean(base_iou, axis=1), color="red", marker='o')
    plt.scatter([x for x in range(len(base_iou))], np.mean(new_iou, axis=1), color="green", marker='o')
    plt.show()

    plt.scatter([x for x in range(len(base_iou))], np.mean(base_acc, axis=1), color="red", marker='o')
    plt.scatter([x for x in range(len(base_iou))], np.mean(new_acc, axis=1), color="green", marker='o')
    plt.show()
    
    