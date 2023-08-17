使用说明
若使用中出现问题，查看出错位置
单图配置，使用默认图片进行配置，选取方式会导致问题
所有配置在默认情况下均能正常运行

1. 先在数据增强的单图配置中设置变换效果

变换设置方式：
变换效果可以通过勾选右侧的增强方式选取，有多种类型以及每种增强方式的选取方式，每种增强方式都有对应的参数进行设置
Oneof:多种方式其中之一
Compose：多种方式都有

2. 确定好增强方式后，点击添加变换，之后应用图像/切换，就可预览效果
3. 当都确认好后增强方式，切换到批量配置
4. 设置需要增强的数据集，目录结构为
图片文件夹  /path/*.png
标注文件夹  /path/*.txt
标注格式默认即可
5. 输出目录需要已存在，点击生成即可



按钮说明：
数据增强 - 单图配置：
清空勾选：去除勾选的变换
添加变换：将勾选的变换添加到图像中，这一步图片不会有反应
应用图像/切换：将变换的方法添加到数据中并进行展示
清空变换：去除变换
0 ： 删除右下方文本框中某一行变换，一种变换一行，支持-1
删除变换：配合上面的数值输入框
导出配置：将配置文件导出到文件
导入配置：将配置文件导入到文件，默认路径


数据处理-easydl：
设置cookie文件：在输入框中输入easydl的cookie字符串或cookie的txt文件路径
下载：设置完上面的内容后，点击下载即可

数据处理-数据转化： 对easydl得到的数据进行格式转换
类别： 用来生成对应的索引数值




数据处理-单张图片标注： 
图片路径：在通用配置中图片输入设置
保存路径：在通用配置中标签输出设置
类别：所有的类 单个图片的类， 类之间使用空格隔开
转换：执行功能

数据处理-多张图片标注： 
【注意】：多张图片中的每张图片都为单类别
        目标图大小不能大于背景图

图片输入路径：在通用配置中图片输入设置
标签输入路径：在通用配置中标签输入设置
数据格式： 通用配置配置
生成数量：图片合成数量数目
图片个数： 一张图有几个目标，一个范围，随机取值
重叠阈值：每个目标重叠的比例
背景图目录：设置背景图目录
图片输出路径：在通用配置中图片输出设置
标签输出路径：在通用配置中标签输出设置
文件名：通用配置，设置
保存路径：在通用配置中标签输出设置


支持的类型



~~AdvancedBlur~~
~~Blur~~
~~Defocus~~
~~GaussianBlur~~
~~GlassBlur~~
~~MedianBlur~~
~~MotionBlur~~
~~RandomBrightnessContrast~~
~~RandomFog~~
~~RandomGamma~~
~~RandomGravel~~
~~RandomRain~~
~~RandomShadow~~
~~RandomSnow~~
~~RandomSunFlare~~
~~RandomToneCurve~~
~~ZoomBlur~~

CLAHE
ChannelDropout
ChannelShuffle
ColorJitter
Downscale
Emboss
Equalize
FDA
FancyPCA
FromFloat
GaussNoise
HistogramMatching
HueSaturationValue
ISONoise
ImageCompression
InvertImg
MultiplicativeNoise
Normalize
PixelDistributionAdaptation
Posterize
RGBShift
RingingOvershoot
Sharpen
Solarize
Spatter
Superpixels
TemplateTransform
ToFloat
ToGray
ToRGB
ToSepia
UnsharpMask





Crop
Flip
NoOp
Affine

Lambda
Resize
Rotate
Transpose
CenterCrop
CropAndPad
SafeRotate
PadIfNeeded
Perspective
PixelDropout

VerticalFlip
GridDistortion
HorizontalFlip
LongestMaxSize
PiecewiseAffine
SmallestMaxSize
ElasticTransform
ShiftScaleRotate
OpticalDistortion
CropNonEmptyMaskIfExists

RandomCrop
RandomScale
RandomRotate90
RandomSizedCrop
RandomResizedCrop
RandomCropNearBBox
BBoxSafeRandomCrop
RandomCropFromBorders
RandomSizedBBoxSafeCrop
