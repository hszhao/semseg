#!/usr/bin/env python
# Martin Kersner, m.kersner@gmail.com
# 2016/01/25

from __future__ import print_function
import os
import sys
from skimage.io import imread, imsave
from utils import convert_from_color_segmentation


def main():
    ##
    ext = '.png'
    ##

    path, txt_file, path_converted = process_arguments(sys.argv)

    # Create dir for converted labels
    if not os.path.isdir(path_converted):
        os.makedirs(path_converted)

    with open(txt_file, 'rb') as f:
        for img_name in f:
            img_base_name = img_name.strip()
            img_base_name = str(img_base_name, encoding='utf-8')
            print(path)
            print(img_base_name)
            img_name = os.path.join(path, img_base_name) + ext
            img = imread(img_name)

            if (len(img.shape) > 2):
                img = convert_from_color_segmentation(img)
                imsave(os.path.join(path_converted, img_base_name) + ext, img)
            else:
                print(img_name + " is not composed of three dimensions, therefore "
                                 "shouldn't be processed by this script.\n"
                                 "Exiting.", file=sys.stderr)

                exit()


def process_arguments(argv):
    if len(argv) != 4:
        help()

    path = argv[1]
    list_file = argv[2]
    new_path = argv[3]

    return path, list_file, new_path


def help():
    print('Usage: python convert_labels.py PATH LIST_FILE NEW_PATH\n'
          'PATH points to directory with segmentation image labels.\n'
          'LIST_FILE denotes text file containing names of images in PATH.\n'
          'Names do not include extension of images.\n'
          'NEW_PATH points to directory where converted labels will be stored.'
          , file=sys.stderr)

    exit()


if __name__ == '__main__':
    main()