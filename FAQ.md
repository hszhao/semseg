# FAQ

This document covers frequently asked questions.

#### Q: Cannot meet the hardware requirement?

**A:** Some choices: 1. Reduce the crop size. 2. Reduce the batch size. 3. Fix BN parameters (scale and shift) for pre-trained models and do not add new BN layers in the network (same as MaskRCNN does). In this case, you may need to modify some code and then train on one GPU is fine. These solutions may harm the performance in a certain degree.

#### Q: Dataset format?

**A:** Assuming `C` as number of classes in the semantic segmentation dataset (e.g., 150 for ADE20K, 21 for PSACAL VOC2012 and 19 for Cityscapes), then valid label ids are from `0` to `C-1`. And we tend to set the ignore label as 255 where loss calculation will be ignored and no penalty will be given on the related ground truth regions. If original ground truths ids are not in needed format, you may need to do label id mapping (e.g, ADE20K original ids are 0-150 where 0 stands for void, original Cityscapes labels also need to do [mapping](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py)).

#### Q: Training on customized dataset?

**A:** Prepare the `$DATASET$_colors.txt` and `$DATASET$_names.txt` accordingly. Get the training/testing ground truths and lists ready.
