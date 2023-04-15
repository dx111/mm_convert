# convert_to_mm
- [convert\_to\_mm](#convert_to_mm)
  - [安装](#安装)
  - [快速开始](#快速开始)
    - [caffe模型转mm](#caffe模型转mm)
    - [onnx模型转mm](#onnx模型转mm)
    - [pytorch模型转mm](#pytorch模型转mm)
    - [tensorflow pb模型转mm](#tensorflow-pb模型转mm)
  - [参数介绍](#参数介绍)
    - [-f(--framework) 原模型框架](#-f--framework-原模型框架)
    - [-m(--model) 原模型](#-m--model-原模型)
    - [-o(--output\_model) 输出模型名](#-o--output_model-输出模型名)
  - [优化选项](#优化选项)
    - [固定模型的输入shape](#固定模型的输入shape)
    - [设置模型运行的设备](#设置模型运行的设备)
    - [设置精度和量化](#设置精度和量化)
      - [使用图片量化](#使用图片量化)
      - [导出自定义数据量化](#导出自定义数据量化)
      - [使用内置数据集量化](#使用内置数据集量化)
    - [改变输入和输出布局](#改变输入和输出布局)
    - [首层做数据归一化](#首层做数据归一化)
    - [交换模型的BR通道](#交换模型的br通道)
    - [添加目标检测大算子](#添加目标检测大算子)
    - [调试参数](#调试参数)


## 安装
```bash
pip install mm_convert
```

## 快速开始
⚠注意：以下命令生成的模型，性能不能达到最佳，建议增加优化参数
### caffe模型转mm
```bash
mm_convert \
    --framework caffe \
    --proto resnet50.prototxt \
    --model resnet50.caffemodel \
    --output_model caffe_resnet50_model
```
### onnx模型转mm
```bash
mm_convert \
    -f onnx \
    --model densenet-12.onnx \
    --output_model onnx_densenet121_model
```
### pytorch模型转mm
```bash
mm_convert \
    -f pt \
    --model resnet50_jit.pt \
    --output_model pt_resnet50_model
```
### tensorflow pb模型转mm
```bash
mm_convert \
    -f tf \
    --model resnet50_v1.pb \
    --output_model pt_resnet50_model \
    --tf_graphdef_inputs input:0 \
    --tf_graphdef_outputs resnet_v1_50/predictions/Softmax:0
```


## 参数介绍
### -f(--framework) 原模型框架
原模型的框架，caffe，onnx，pytorch，tensorflow，pytorch可以使用简写pt，tensorflow可以使用简写tf
example:
```bash
--framework caffe
-f onnx
-f pt
-f tf
```

### -m(--model) 原模型
原模型的模型文件，对于不同的框架，指代的文件不同，对于caffe模型，还需要使用参数--proto指定prototxt模型文件
```bash
# for caffe
--model resnet50.caffemodel --proto resnet50.prototxt
# for onnx
--model resnet50.onnx
# for pytorch
-m resnet50_jit.pt
# for tensorflow
-m resnet50.pb
```

### -o(--output_model) 输出模型名
生成mm模型的输出目录，当未指定的时候，默认生成原模型名字相同，增加.mm后缀，例如models/resnet50.onnx，会生成resnet50.onnx.mm的模型

## 优化选项

### 固定模型的输入shape
使用--input_shapes设置网络输入的shape，会自动设置网络的输入shape不可变，提升网络的性能，如需生成可变模型，请设置graph_shape_mutable=true    

输入1的shape是1,3,224,224
```bash
--input_shapes 1,3,224,224
```

输入1的shape是1,128, 输入2的shape是1,256，输入3的shape是1,123
```bash
--input_shapes 1,128 1,256 1,123
```

输入1的shape是1,3,224,224，并指定输入shape可变
```bash
--input_shapes 1,3,224,224 \
--graph_shape_mutable true
```

### 设置模型运行的设备
通过指定archs，指定生成mlu370或者3226的模型，并指定多核优化,使用方法     
指定生成3226的模型
```bash
--archs mtp_322
```

指定生成3226和370的模型
```bash
--archs mtp_322 mtp_372
```

指定生成3226模型，和370 8核模型
```bash
--archs mtp_322 mtp_372:8
```


指定生成3703226模型，和370 6核和8核模型
```bash
--archs mtp_322 mtp_372:6,8
```
对于3226(单核)，无须设置多核优化，对于370-s4(6核)，建议设置mtp_372:6，对于370-s4(8核)，建议设置mtp_372:8

### 设置精度和量化
使用--precision 参数可以设置模型的精度，当涉及量化时，需要指定量化数据，量化数据的数据分布应与真实的数据分布一致
| 精度 | 介绍 |
| ------ | ------ |
| qint8_mixed_float16 | conv,matmul类算子使用qint8,其他算子使用float16 |
| qint8_mixed_float32 | conv,matmul类算子使用qint8,其他算子使用float32 |
| qint16_mixed_float16 | conv,matmul类算子使用qint16,其他算子使用float16 |
| qint16_mixed_float32 | conv,matmul类算子使用qint16,其他算子使用float32 |
| force_float16 | 网络中所有算子使用float16 |
| force_float32 | 网络中所有算子使用float32 |

设置生成模型的精度为 force_float16
```bash
--precision force_float16
```

设置生成模型的精度为 force_float32
```bash
--precision force_float32
```

#### 使用图片量化
当使用图片量化时，需要指定以下参数
| 参数         | 默认值       | 说明 |
| ----         | ----        | ---- |
| image_dir    | 内置图片目录 | 量化图片所在的文件在                                     |
| image_color  | rgb         | 训练模型所用的图片的颜色空间, rgb或bgr                    |
| image_mean   | 0.0         | 图片的均值, img = img - mean                            |
| image_std    | 1.0         | 图片的方差, img = img / std                             |
| image_scale  | 1.0         | 缩放系数   img = img * scale，此操作在减均值，除方差之前  |

例子1：模型A训练用的是bgr的图片, 预处理没有减均值，只除了255(img/=255.0)，用的img_data1目录下的图片，此时参数配置如下
```bash
--precision qint8_mixed_float16 \
--image_dir img_data1 \
--image_color bgr \
--image_std 255.0,255.0,255.0 
```

例子2：模型B训练用的是rgb的图片，预处理经过了transforms.Normalize(等效于img*=(1/255.0)),再减去均值(img-=[0.485,0.456,0.406]),再除以标准差(img/=[0.229,0.224,0.225]),此时参数配置如下
```bash
--precision qint8_mixed_float16 \
--image_dir sample_data/voc \
--image_color rgb \
--image_mean 0.485,0.456,0.406 \
--image_std 0.229,0.224,0.225 \
--image_scale 1/255.0,1/255.0,1/255.0
```

模型c训练用到了两张图片，第一张图片预处理和模型A相同，第二张图片预处理和模型B相同，精度设置为qint16_mixed_float16
```bash
--precision qint16_mixed_float16 \
--image_dir sample_data/imagenet \
--image_color bgr rgb \
--image_mean 0.0,0.0,0.0 0.485,0.456,0.406 \
--image_std 255.0,255.0,255.0 0.229,0.224,0.225 \
--image_scale 1.0,1.0,1.0 1/255.0,1/255.0,1/255.0
```

#### 导出自定义数据量化
在原始模型的推理代码中，使用以下代码保存数据,使用add添加的数据需为numpy数据
```bash
from mm_convert import Record
...some code ...

record = Record()

for input_data in datas:
    record.add(input_data.numpy())
    model.predict(input_data)
record.save("calibrate_data")
```

保存量化数据后，进行量化
```bash
--load_data_func load_calibrate_data \
--calibrate_data_file calibrate_data
```

#### 使用内置数据集量化
使用bert模型时，指定
```bash
--load_data_func load_squad
```
使用该参数会调用dataloader.py文件中的load_squad函数方法，返回量化数据集，也可以参数该函数，制作自己的数据集


### 改变输入和输出布局
图片加载后一般的数据格式是(h,w,c)，增加batch后是(n,h,w,c),对于网络的输入是(n,c,h,w),图片需要transpose(n,h,w,c)->(n,c,h,w)后，才能进行推理，通过设置参数input_as_nhwc，将网络的输入转变为nhwc后，可免去图片的transpose      

将输入1从nchw改成nhwc
```bash
--input_as_nhwc true \
```

输入1不变，将输入2从nchw改成nhwc
```bash
--input_as_nhwc false true \
```

将输出1和输出2从nchw改成nhwc
```bash
--output_as_nhwc false true
```

### 首层做数据归一化
对于首层是conv的网络，可以设置insert_bn，代替预处理中的归一化操作，设置insert_bn之后，无须再做减均值除标准差的归一化操作，输入的数据类型也会变成uint8(fp32->uint8,减少3/4的数据量)，注意此参数的开启依赖与正确的设置了 image_mean,image_std,image_scale参数，此参数在精度和量化校准提及，不在赘述    
输入1开启insert_bn
```bash
--insert_bn true \
```

输入1不变，输入2开启insert_bn
```bash
--insert_bn false true \
```

### 交换模型的BR通道
对于一些已经训练好的模型，训练时采用的bgr或者rgb的数据，推理时图片需要对图片，进行rbg2bgr或者bgr2rgb的转换，此转换浪费了时间，可对模型进行更改，交换权值中的B,R通道，使其接收另一种颜色空间的图片
```bash
--model_swapBR true
```

### 添加目标检测大算子
对于yolo ssd 之类的网络，使用大算子代替原生的检测层，可大幅提升性能    
yolov3 检测层的配置，
```bash
--add_detect  true \
--detect_bias 116,90,156,198,373,326,30,61,62,45,59,119,10,13,16,30,33,23 \
--detect_algo yolov3 
```

yolov5 配置如下
```bash
--add_detect true \
--detect_bias 10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326 \
--detect_algo yolov5
```
注意事项：    
默认情况下，会在网络的最后增加目标检测算子，因此需要去掉网络原始的检测层，仅保留网络最后的特征图   
1. caffe 模型    
caffe 模型修改prototxt文件，去掉DetectionOutput层即可    
2. tensorflow pb模型
tensorflow的pb格式模型，可以指定参数--tf_graphdef_outputs conv_lbbox/Conv2D:0 conv_mbbox/Conv2D:0 conv_sbbox/Conv2D:0，指定网络的输出为卷积后的特征图，name需要根据情况更改
3. pytorch 模型
pytorch模型需要在jit.trace模型的时候，在代码中修改，将检测层的的输入直接return，不经过检测层

参数
1.detect_add_permute_node    
detect层需要nhwc的输入，当特征图是nchw时(caffe, pytorch),需要添加此参数，将特征图permute为nhwc    

2.detect_algo     
目标检测的算子，支持的参数有yolov2 yolov3 yolov4 yolov5 fasterrcnn ssd refinedet    

3.detect_bias    
anchor值，anchor值应该分组，由大到小排列，yolov3有三组anchor，大anchor值116,90,156,198,373,326， 中anchor值30,61,62,45,59,119，小anchor值10,13,16,30,33,23    
yolov3的anchor值设置,此值为默认值
```bash
--detect_bias 116,90,156,198,373,326,30,61,62,45,59,119,10,13,16,30,33,23
```
4.detect_num_class    
目标检测的类别数，默认80    

5.detect_conf    
目标检测框的阈值，默认0.0005     

6.detect_nms     
目标检测，nms的阈值，默认0.45    

7.detect_image_shape    
目标检测图片的尺寸,未设置会根据网络的输入0和参数image_size进行推到，无法推导则设置为416,416

### 调试参数
保存模型build期间，图的结构
```bash
--print_ir true
```
