# convert_to_mm
- [convert\_to\_mm](#convert_to_mm)
  - [安装](#安装)
  - [快速开始](#快速开始)
  - [模型性能优化](#模型性能优化)
    - [模型固定shape](#模型固定shape)
  - [mm\_convert参数介绍](#mm_convert参数介绍)
    - [通用参数](#通用参数)
      - [-f(--framework) 原模型框架](#-f--framework-原模型框架)
      - [-m(--model) 原模型](#-m--model-原模型)
      - [-o(--output\_model) 输出的mm模型名](#-o--output_model-输出的mm模型名)
      - [--input\_shapes 输出的shape](#--input_shapes-输出的shape)


## 安装
```bash
pip install mm_convert
```

## 快速开始
caffe模型转mm
```bash
mm_convert \
--framework caffe \
--proto resnet50.prototxt \
--model resnet50.caffemodel \
--output_model caffe_resnet50_model
```
onnx模型转mm
```bash
mm_convert \
-f onnx \
--model densenet-12.onnx \
--output_model onnx_densenet121_model
```
pytorch模型转mm
```bash
mm_convert \
-f pt \
--model resnet50_jit.pt \
--output_model pt_resnet50_model
```
tensorflow pb模型转mm
```bash
mm_convert \
-f tf \
--model resnet50_v1.pb \
--output_model pt_resnet50_model \
--tf_graphdef_inputs input:0 \
--tf_graphdef_outputs resnet_v1_50/predictions/Softmax:0
```

## 模型性能优化
### 模型固定shape


## mm_convert参数介绍
### 通用参数
#### -f(--framework) 原模型框架
原模型的框架，caffe，onnx，pytorch，tensorflow，pytorch可以使用简写pt，tensorflow可以使用简写tf
example:
```bash
--framework caffe
-f onnx
-f pt
-f tf
```

#### -m(--model) 原模型
原模型的模型文件，对于不同的框架，指代的文件不同,对于caffe，还需要指定--proto
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
#### -o(--output_model) 输出的mm模型名
#### --input_shapes 输出的shape


<!-- ### archs
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
对于3226(但核)，无须设置多核优化，对于370-s4(6核)，建议设置mtp_372:6，对于370-s4(8核)，建议设置mtp_372:8
### input_shapes
设置网络input_shape，会自动设置网络的输入shape不可变，大幅提升网络的性能，如需生成可变模型，请设置graph_shape_mutable=true    

指定输入1的shape是1,3,224,224
```bash
--input_shapes 1,3,224,224
```

指定输入1的shape是1,128, 输入2的shape是1,256，输入3的shape是1,123
```bash
--input_shapes 1,128 1,256 1,123
```

指定输入1的shape是1,3,224,224，并指定输入shape可变
```bash
--input_shapes 1,3,224,224 \
--graph_shape_mutable true
```

### 设置精度和量化

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

当设置精度为量化精度时，需要指定量化数据，量化数据的数据分布应与真实的数据分布一致，内置了量化数据的加载函数，默认是加载图片数据，图片的路径由参数image_dir指定，会自动搜索此路径下的图片文件    
不同输入的image_mean,image_std,image_scale以空格隔开，单个输入不同通道的值以逗号(,)隔开，支持简单的算术表达式，1/255.0之类的值    
模型A训练用的是bgr的图片, 预处理没有减均值，只除了255(img/=255.0)，此时参数配置如下
```bash
--precision qint8_mixed_float16 \
--image_dir sample_data/imagenet \
--image_color bgr \
--image_std 255.0,255.0,255.0 
```


模型B训练用的是rgb的图片，预处理经过了transforms.Normalize(等效于img/=255.0),再减去均值(img-=[0.485,0.456,0.406]),再除以标准差(img/=[0.229,0.224,0.225]),此时参数配置如下
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

### insert_bn
对于首层是conv的网络，可以设置insert_bn，代替预处理中的归一化操作，设置insert_bn之后，无须再做减均值除标准差的归一化操作，输入的数据类型也会变成uint8(fp32->uint8,减少3/4的数据量)，注意此参数的开启依赖与正确的设置了 image_mean,image_std,image_scale参数，此参数在精度和量化校准提及，不在赘述    
输入1开启insert_bn
```bash
--insert_bn true \
```

输入1不变，输入2开启insert_bn
```bash
--insert_bn false true \
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

#### detect 参数
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

### bgr模型转rgb模型
对于一些已经训练好的模型，训练时采用的bgr或者rgb的数据，推理时图片需要对图片，进行rbg2bgr或者bgr2rgb的转换，此转换浪费了时间，可对模型进行更改，交换权值中的B,R通道，使其接收另一种颜色空间的图片
```bash
--model_swapBR true
```

### 调试参数
保存模型build期间，图的结构
```bash
--print_ir true
``` -->
