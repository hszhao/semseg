# The Details of preparing for VOC 2012 Dataset

## 1.1 Download these dataset

- [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/)
- [Semantic Boundaries Dataset and Benchmark](http://home.bharathh.info/pubs/codes/SBD/download.html)

After you have downloaded these files, you will get 2 file named `VOCtrainval_11-May-2012.tar`, `benchmark.tgz` respectively. And the next step is to augment VOC dataset with SBD.

## 1.2 Augment VOC Dataset with SBD
As you can see `dataset/voc2012/train.txt` file, each row is composed by 2 columns. The 1st column is `raw picture`, another column is `target mask`(also named ground truth). It is strange that where the `SegmentationClassAug` fold is? I extract 2 dataset and don't find this folder.

So I will solve this problem in this section. If you also meet this error, hope this article is helpful for you.

### 1.2.1 数据集结构(The Structure of Dataset)

```js
VOCdevkit/
└── VOC2012
    ├── Annotations
    ├── ImageSets
    ├── JPEGImages
    ├── SegmentationClass
    └── SegmentationObject
```
- Annotations: 这个文件夹内主要存放了数据的标签，里面包含了每张图片的bounding box信息，主要用于目标检测，这里不做过多介绍。
- ImageSets: ImageSets中的Segmentation目录下存放了用于分割的train, val, trainval数据集的索引。
- JPEGImages: 这里存放的就是JPG格式的原图，包含17125张彩色图片，但只有一部分(2913张)是用于分割的。
- SegmentationClass: 语义分割任务中用到的label图片，PNG格式，共2913张，与原图的每一张图片相对应。
- SegmentationObject: 实例分割任务用到的label图片，在语义分割中用不到，这里不详解介绍。

```
benchmark_RELEASE/
├── benchmark_code_RELEASE
│   ├── *.m
├── BharathICCV2011.pdf
├── dataset
│   ├── cls
│   ├── img
│   ├── inst
│   ├── train.txt
│   └── val.txt
└── README
```
增强版的VOC跟原版没什么区别，只是增加了数据。下面还是一一介绍下载的文件夹中每个文件的作用。

- img: 增强版的原图，共11355张图片
- cls: 用于语义分割的label，共11355个.mat文件，每个.mat文件对应一张原图
- inst: 用于实例分割的label，也是11355个.mat文件
- tools: 3个用于数据转换的脚本，在后面数据转换时用到
- train.txt: 训练集索引
- val.txt: 验证集索引
    
### 1.2.2 数据转换 (Transition of data)
大多数paper中使用的数据集是以上两个数据集的融合，因此，在这部分将要介绍如何融合以上两个数据集来训练。

Step 1. 数据转换

由于pascal voc 2012增强版的数据集的label是.mat格式的文件，需要将其转换为.png格式的图片。转化后的图片是8-bit的灰度图。

```js
1) 在benchmark_RELEASE的dataset目录下，创建cls_png
darling@G7:~/dataset/pascal_voc/benchmark_RELEASE/dataset$ mkdir cls_png
2) 在benchmark_RELEASE的dataset目录下，执行mat2png.py脚本(由于该目录没有，请拷贝本项目的voc_tool目录)
darling@G7:~/dataset/pascal_voc/benchmark_RELEASE/dataset$ python tools/mat2png.py cls cls_png
```

 原始pascal voc 2012数据集中label为4通道RGB图像，为了统一，我们也将其转化为8-bit的灰度png图像。
```js
1) 在VOCdevkit/VOC2012下创建SegmentationClass_1D文件夹用于存放转化后的图片
darling@G7:~/dataset/pascal_voc/VOCdevkit/VOC2012$ mkdir SegmentationClass_1D
2) 在VOCdevkit/VOC2012下执行转换操作
darling@G7:~/dataset/pascal_voc/VOCdevkit/VOC2012$ python tools/convert_labels.py SegmentationClass ImageSets/Segmentation/trainval.txt SegmentationClass_1D
```

Step 2. 数据合并
```js
1) 把SegmentationClass_1D中生成的图片都拷贝到cls_png目录
darling@G7:~/dataset/pascal_voc/VOCdevkit/VOC2012$ cp SegmentationClass_1D/* ~/dataset/pascal_voc/benchmark_RELEASE/dataset/cls_png/
2) 在VOCdevkit/VOC2012/JPEGImages下的图片，都拷贝到benchmark_RELEASE/dataset/img/目录
darling@G7:~/dataset/pascal_voc/VOCdevkit/VOC2012$ cp JPEGImages/* ~/dataset/pascal_voc/benchmark_RELEASE/dataset/img/
```

Step 3. 重整目录格式
```js
1) 目录改名
darling@G7:~/dataset/pascal_voc/benchmark_RELEASE/dataset$ mv img JPEGImages
darling@G7:~/dataset/pascal_voc/benchmark_RELEASE/dataset$ mv cls_png SegmentationClassAug
1) 目录移到上级目录
darling@G7:~/dataset/pascal_voc/benchmark_RELEASE/dataset$ mv JPEGImages SegmentationClassAug ../../
```

最后经过清理无用的文件，得到的目录结构如下：

```js
.
├── JPEGImages
├── SegmentationClassAug
├── VOCtrainval_11-May-2012.tar
└── benchmark.tgz
```

其中主要用的只有`JPEGImages`和`SegmentationClassAug`目录。

## 1.3 训练的注意事项
- `tool/train.sh`中，第10行默认指定的conda的`test`运行环境，注意更改！
- `config`中包含数据集的目录设置！
- `config/voc2012/*` 在配置数据集时，`data_root`参数请配到包含`JPEGImages`与`SegmentationClassAug`的目录，本文直接通过软链方式衔接。
- `initmodel`文件的下载，请访问[initmodel.tar.gz](https://download.csdn.net/download/u010516952/15991282)下载。