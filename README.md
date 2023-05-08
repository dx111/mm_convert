# 1. convert_to_mm
- [1. convert\_to\_mm](#1-convert_to_mm)
  - [1.1. 安装](#11-安装)
  - [1.2. 快速开始](#12-快速开始)
    - [1.2.1. caffe模型转mm](#121-caffe模型转mm)
    - [1.2.2. onnx模型转mm](#122-onnx模型转mm)
    - [1.2.3. pytorch模型转mm](#123-pytorch模型转mm)
    - [1.2.4. tensorflow pb模型转mm](#124-tensorflow-pb模型转mm)
  - [1.3. 参数介绍](#13-参数介绍)
  - [1.4. 参数用法](#14-参数用法)
    - [1.4.1. 原模型框架](#141-原模型框架)
    - [1.4.2. 原模型文件路径](#142-原模型文件路径)
    - [1.4.3. 设置输出模型名](#143-设置输出模型名)
    - [1.4.4. 固定模型的输入shape](#144-固定模型的输入shape)
    - [1.4.5. 设置模型运行的设备](#145-设置模型运行的设备)
    - [1.4.6. 设置精度](#146-设置精度)
    - [1.4.7. 使用图片量化](#147-使用图片量化)
    - [1.4.8. 导出自定义数据量化](#148-导出自定义数据量化)
    - [1.4.9. 使用内置数据集量化](#149-使用内置数据集量化)
    - [1.4.10. 改变输入和输出布局](#1410-改变输入和输出布局)
    - [1.4.11. 首层做数据归一化](#1411-首层做数据归一化)
    - [1.4.12. 交换模型的BR通道](#1412-交换模型的br通道)
    - [1.4.13. 添加目标检测大算子](#1413-添加目标检测大算子)
    - [1.4.14. 调试参数](#1414-调试参数)
  - [1.5. 例子](#15-例子)
    - [1.5.1. yolov7](#151-yolov7)


## 1.1. 安装
```bash
pip install mm_convert
```

## 1.2. 快速开始
⚠注意：以下命令生成的模型，性能不能达到最佳，建议增加优化参数
### 1.2.1. caffe模型转mm
```bash
mm_convert \
    --framework caffe \
    --proto resnet50.prototxt \
    --model resnet50.caffemodel \
    --output_model caffe_resnet50_model
```
### 1.2.2. onnx模型转mm
```bash
mm_convert \
    -f onnx \
    --model densenet-12.onnx \
    --output_model onnx_densenet121_model
```
### 1.2.3. pytorch模型转mm
```bash
mm_convert \
    -f pt \
    --model resnet50_jit.pt \
    --output_model pt_resnet50_model
```
### 1.2.4. tensorflow pb模型转mm
```bash
mm_convert \
    -f tf \
    --model resnet50_v1.pb \
    --output_model pt_resnet50_model \
    --tf_graphdef_inputs input:0 \
    --tf_graphdef_outputs resnet_v1_50/predictions/Softmax:0
```
更多示例请参开exmples目录下的脚本

## 1.3. 参数介绍
| 通用参数         | 默认值   | 示例     | 说明 |
| ----         | ----    | ----  | ---- |
| framework    | None    | --framework onnx | 原模型的框架                 |
| model    | None    | --model xxx_model | 原模型模型文件                          |
| proto    | None    | --proto xxx.prototxt | caffe的prototxt文件 |
| output_model    | None    | --output_model model.mm | 参数为空时，生成与原模型名相同，后缀增加.mm的模型 |
| archs    | None    | --archs mtp_372 | 设置模型运行的设备 [arch的设置](#145-设置模型运行的设备) |
| input_shapes    | None    | --input_shapes 1,3,224,224 | 设置模型的输入形状 |
| graph_shape_mutable    | None    | --graph_shape_mutable true | 输入支持可变 |


| 量化相关参数         | 默认值   | 示例     | 说明 |
| ----         | ----    | ----  | ---- |
| precision    | None    | --precision q8 | 模型的精度,若使用量化精度，则需指定量化参数，[支持的精度](#146-设置精度) |
| 使用图片量化 
| image_dir    | None    | --image_dir image  | 图片文件夹,为None时使用内置图片                 |
| image_color  | rgb     | --image_color bgr   | 训练模型所用的图片的颜色空间, rgb或bgr          |
| image_mean   | 0.0     | --image_mean 0.485,0.456,0.406    | 图片的均值, img = img - mean          |
| image_std    | 1.0     | --image_std 255.0,255.0,255.0   | 图片的方差, img = img / std           |
| image_scale  | 1.0     | --image_scale 1/255.0,1/255.0,1/255.0 | 缩放系数   img = img * scale  |


| 模型优化相关         | 默认值   | 示例     | 说明 |
| ----         | ----    | ----  | ---- |
| input_as_nhwc    | None    | --input_as_nhwc true | 将模型输入的layout由nchw转为nhwc |
| output_as_nhwc    | None    | --output_as_nhwc true | 将模型输出的layout由nchw转为nhwc |
| insert_bn    | None    | --insert_bn true | 模型做输入数据归一化，该参数依赖--image_mean image_std image_scale |
| model_swapBR    | None    | --model_swapBR true | rgb模型和bgr模型互转 |

| 目标检测算子相关参数         | 默认值   | 示例     | 说明 |
| ----         | ----    | ----  | ---- |
| add_detect  | false    | --add_detect true | 向网络增加目标检测大算子 |
| detect_add_permute_node | true    | --detect_add_permute_node true | 目标检测算子只支持nhwc，使用permute转换layput |
| detect_bias | yolov3的anchor    | --detect_bias 116,90,156,198,373,326,30,61,62,45,59,119,10,13,16,30,33,23 | anchor box |
| detect_num_class | 80    | --detect_num_class 80 | 目标检测的类别数 |
| detect_conf | 0.0005    | --detect_conf 0.3 | 目标检测的置信度 |
| detect_nms  | 0.45    | --detect_conf 0.45 | nms的阈值 |
| detect_image_shape  | None    | --detect_image_shape 640,640 | 目标检测图片的shape，默认根据input shape推导 |

## 1.4. 参数用法
### 1.4.1. 原模型框架
原模型的框架，caffe，onnx，pytorch，tensorflow，pytorch可以使用简写pt，tensorflow可以使用简写tf
example:
```bash
--framework caffe
-f onnx
-f pt
-f tf
```

### 1.4.2. 原模型文件路径
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

### 1.4.3. 设置输出模型名
```bash
--output_model model.mm
```
生成mm模型的输出目录，当未指定的时候，默认生成原模型名字相同，增加.mm后缀，例如models/resnet50.onnx，会生成resnet50.onnx.mm的模型

### 1.4.4. 固定模型的输入shape
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

### 1.4.5. 设置模型运行的设备
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

### 1.4.6. 设置精度
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

### 1.4.7. 使用图片量化
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

### 1.4.8. 导出自定义数据量化
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

### 1.4.9. 使用内置数据集量化
使用bert模型时，指定
```bash
--load_data_func load_squad
```
使用该参数会调用dataloader.py文件中的load_squad函数方法，返回量化数据集，也可以参数该函数，制作自己的数据集


### 1.4.10. 改变输入和输出布局
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

### 1.4.11. 首层做数据归一化
对于首层是conv的网络，可以设置insert_bn，代替预处理中的归一化操作，设置insert_bn之后，无须再做减均值除标准差的归一化操作，输入的数据类型也会变成uint8(fp32->uint8,减少3/4的数据量)，注意此参数的开启依赖与正确的设置了 image_mean,image_std,image_scale参数，此参数在精度和量化校准提及，不在赘述    
输入1开启insert_bn
```bash
--insert_bn true \
```

输入1不变，输入2开启insert_bn
```bash
--insert_bn false true \
```

### 1.4.12. 交换模型的BR通道
对于一些已经训练好的模型，训练时采用的bgr或者rgb的数据，推理时图片需要对图片，进行rbg2bgr或者bgr2rgb的转换，此转换浪费了时间，可对模型进行更改，交换权值中的B,R通道，使其接收另一种颜色空间的图片
```bash
--model_swapBR true
```

### 1.4.13. 添加目标检测大算子
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

### 1.4.14. 调试参数
保存模型build期间，图的结构
```bash
--print_ir true
```

## 1.5. 例子

### 1.5.1. yolov7

1 使用官方仓库的的python程序将yolov7转为onnx模型
```bash
python export.py --weights yolov7-tiny.pt --grid --end2end --simplify \
        --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
```

2 执行转换     
请根据实际情况修改参数
```
mm_convert \
    -f onnx \
    --model yolov7/yolov7.onnx \
    --output_model onnx_yolov7 \
    --archs mtp_372.41 \
    --input_shapes 1,3,640,640 \
    --input_as_nhwc true \
    --insert_bn true \
    --precision qint8_mixed_float16 \
    --image_color rgb \
    --image_scale 1/255.0,1/255.0,1/255.0 \
    --add_detect true \
    --detect_bias 12,16,19,36,40,28,36,75,76,55,72,146,142,110,192,243,459,401 \
    --detect_image_shape 640,640 \
    --detect_algo yolov5 
```
注意事项：
1. 该脚本使用了yolov5的目标检测大算子替换原生的检测层，如果不需要，设置--add_detect false
2. 默认的detect_conf是0.0005，detect_nms是0.45，类别数为80，请根据实际情况修改

常见问题：
1. channel不匹配：      
默认的类别数是80，255(channel)=(80(类别数)+1(分数)+4(坐标))*3(三个检测头)，设置的类别数和特征图的channel数不匹配，会报此错误
