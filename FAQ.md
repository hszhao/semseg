# FAQ

This document covers frequently asked questions.

#### Q: Hardware requirement?

**A:** For training, ADE20K and PSACAL VOC 2012 with crop size 473\*473 and batch size 16 require 4*12G GPUs (ResNet50/101 based), and Cityscapes with crop size 713\*713 and batch size 16 requires 8\*12G GPUs   (ResNet50/101 based). A workstation with 8\*12G GPUs can run all experiments efficiently. For testing, one GPU with 4GB is enough.

#### Q: Cannot meet the hardware requirement?

**A:** Some choices: 1. Reduce the crop size. 2. Reduce the batch size. 3. Fix BN parameters (scale and shift) for pre-trained models and do not add new BN layers in the network (same as MaskRCNN does). In this case, you may need to modify some code and then train on one GPU is fine. These solutions may harm the performance in a certain degree.

#### Q: Why PyTorch 1.0.0?

**A:** Mainly for the interface difference of CUDA extensions (`syncbn` for multithreading training in this codebase). PyTorch version <=0.4.1 uses [FFI](https://github.com/pytorch/extension-ffi) that is [not supported](https://github.com/pytorch/extension-ffi/issues/19) after 0.5.0, and now [JIT](https://github.com/pytorch/extension-cpp) is preferred. You need to change the interface of extensions under folder`lib` for adapting to former version like 0.4.1.

#### Q: Synchronized batch normalization?

**A:** Synchronized batch normalization crosses multiple GPUs is important for high-level version tasks especially when single card's batch size is not large enough (effective batch size as 16 is a good choice). Former PSPNet Caffe version uses OpenMPI based implementation. In branch `1.0.0`, for multithreading training, the codebase uses synchronized batch normalization from repo [EncNet](https://github.com/zhanghang1989/PyTorch-Encoding), and for multiprocessing training, [NVIDIA/apex](https://github.com/NVIDIA/apex) is adopted. Another multithreading syncbn module is [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch),  some other multiprocessing syncbn modules are [inplace_abn](https://github.com/mapillary/inplace_abn) and the newly released [official implementation](https://pytorch.org/docs/master/nn.html#torch.nn.SyncBatchNorm) in PyTorch 1.1.0. In branch `master`, only multiprocessing training is supported and the official `nn.SyncBatchNorm` is adopted.

#### Q: Training speed up?

**A:** Two possible choices:

1. Multiprocessing training is highly recommended over multithreading training.
2. Using 1/8 scale ground truth as label guidance, this can slightly speeding up the training and slightly decrease the performance (not as good as 1 scale label guidance is most cases).

#### Q: Pre-trained ImageNet models?

**A:** The provided [ResNet.py](./model/resnet.py) with pre-trained models differ with the official [implementation](https://github.com/hszhao/semseg-dev/blob/master/model/resnet.py) in the input stem where original 7 × 7 convolution is replaced by three conservative 3 × 3 convolutions. This replacement is the same as the models used in original PSPNet [Caffe version](https://github.com/hszhao/PSPNet). The classification accuracy is slightly better than official [models](https://pytorch.org/docs/stable/torchvision/models.html). ResNet50/101/152 comparisons in terms of top1 accuracy: ours vs official = 76.63/78.25/78.59 vs 76.15/77.37/78.31. The pre-trained models have slightly influences on final segmentation models (better). You may have a glance at Sec 4.2 `ResNet Tweaks` in this [paper](https://arxiv.org/pdf/1812.01187.pdf). You can also utilize official released models as initialization and you need to modify files under folder `model` accordingly.

#### Q: Performance difference with original papers?

**A:** Lots of details, some are listed as:

1. Pre-trained models: the used weights are different between this PyTorch codebase and former PSP/ANet Caffe version.
2. Pre-processing of images: this PyTorch codebase follows PyTorch official image pre-processing styles (normalized to 0~1 followed by subtracting `mean` as [0.485, 0.456, 0.406] and divided by `std` as  [0.229, 0.224, 0.225]), while former Caffe version do normalization simply by subtracting image `mean` as [123.68, 116.779, 103.939].
3. Training steps: we use training steps in Caffe version and training epochs in PyTorch for measurement. The transformed optimization steps after conversion is slightly different (e.g., in ade20k 150k with 16 batches equals to 150k*16/20210=119 epochs).
4. SGD optimization difference: see `note` in SGD [implementation](https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html), this difference may has influences on `poly` style learning rate decay especially on the last steps where learning rates are very small.
5. Weight decay on biases, scale and shift of BN in two training settings, see technical reports [1](https://arxiv.org/pdf/1812.01187.pdf), [2](https://arxiv.org/pdf/1807.11205.pdf).
6. Label guidance: former Caffe version mainly uses 1/8 scale label guidance (former `interp` layer in Caffe has only CPU implementation thus we avoid using larger label guidance), the released segmentation models in this repository mainly use full scale label guidance (interpolate the final logits to original crop size for loss calculation instead of feature downsampling size as 1/8).
7. The performance variance for attention based models (e.g., PSANet) is relatively high, this can also be observed in [CCNet](https://github.com/speedinghzl/CCNet). Besides, some low frequent classes (e.g, 'bus' in cityscapes) may also affect the performance a lot.

#### Q: Dataset format?

**A:** Assuming `C` as number of classes in the semantic segmentation dataset (e.g., 150 for ADE20K, 21 for PSACAL VOC2012 and 19 for Cityscapes), then valid label ids are from `0` to `C-1`. And we tend to set the ignore label as 255 where loss calculation will be ignored and no penalty will be given on the related ground truth regions. If original ground truths ids are not in needed format, you may need to do label id mapping (e.g, ADE20K original ids are 0-150 where 0 stands for void, original Cityscapes labels also need to do [mapping](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py)).

#### Q: Training on customized dataset?

**A:** Prepare the `$DATASET$_colors.txt` and `$DATASET$_names.txt` accordingly. Get the training/testing ground truths and lists ready.

#### Q: Other related repository?

**A:** Sorted by platforms, you are welcome to add some more.

- PyTorch: [semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch), [InplaceABN](https://github.com/mapillary/inplace_abn), [EncNet](https://github.com/zhanghang1989/PyTorch-Encoding), [OCNet](https://github.com/PkuRainBow/OCNet.pytorch), [CCNet](https://github.com/speedinghzl/CCNet), [DANet](https://github.com/junfu1115/DANet), [TorcgSeg](https://github.com/ycszen/TorchSeg).

- Tensorflow: [DeepLab](https://github.com/tensorflow/models/tree/master/research/deeplab).

- Caffe: [DeepLab](https://bitbucket.org/deeplab/deeplab-public), [PSPNet](https://github.com/hszhao/PSPNet), [PSANet](https://github.com/hszhao/PSPNet), [ICNet](https://github.com/hszhao/PSPNet).