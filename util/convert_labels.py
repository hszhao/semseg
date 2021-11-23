import cv2
from glob import glob

def convert_mask(annotation_path):
    label = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
    fix_path = annotation_path.replace('_original', '')
    label_fix = label - 1
    label_fix[label_fix == -1] = 255
    cv2.imwrite(fix_path, label_fix)


if __name__ == "__main__":
    for image_path in glob("dataset/ade20k/annotations_original/*/*.png"):
        convert_mask(image_path)
