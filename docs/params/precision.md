---
layout: default
title: 精度与量化
parent: 参数介绍
nav_order: 2
permalink: /params/precision
---



### precision
使用--precision 参数可以设置模型的精度，当涉及量化时，需要指定量化数据，为保证精度，量化数据的数据分布，应与真实的数据分布一致
| 精度 | 参数 | 介绍 |
| ------ | ------ | ------ |
| qint8_mixed_float16 | qint8_mixed_float16, q8_f16, q8 | conv,matmul类算子使用qint8,其他算子使用float16 |
| qint8_mixed_float32 | qint8_mixed_float32 q8_f32 | conv,matmul类算子使用qint8,其他算子使用float32 |
| qint16_mixed_float16 | qint16_mixed_float16, q16_f16, q16 | conv,matmul类算子使用qint16,其他算子使用float16 |
| qint16_mixed_float32 | qint16_mixed_float32, q16_f32 | conv,matmul类算子使用qint16,其他算子使用float32 |
| force_float16 | force_float16, f16 | 网络中所有算子使用float16 |
| force_float32 | force_float32, f32 | 网络中所有算子使用float32 |

设置生成模型的精度为 force_float16
```bash
--precision force_float16
```

设置生成模型的精度为 force_float32
```bash
--precision force_float32
```

### load_image
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