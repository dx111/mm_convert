---
layout: default
title: Home
nav_order: 2
description: "Home"
permalink: /overview
---


## 参数介绍
| 通用         | 默认值   | 示例     | 说明 |
| ----         | ----    | ----  | ---- |
| [framework](#framework)    | None    | --framework onnx | 原模型的框架                 |
| [model](#model)    | None    | --model xxx_model | 原模型模型文件                          |
| [proto](#proto)    | None    | --proto xxx.prototxt | caffe的prototxt文件 |
| [output_model](#output_model)    | None    | --output_model model.mm | 参数为空时，生成与原模型名相同，后缀增加.mm的模型 |
| [archs](#archs)    | None    | --archs mtp_372 | 设置模型运行的设备 |
| [input_shapes](#input_shapes)    | None    | --input_shapes 1,3,224,224 | 设置模型的输入形状 |
| [graph_shape_mutable](#graph_shape_mutable)    | None    | --graph_shape_mutable true | 输入支持可变 |


| 精度         | 默认值   | 示例     | 说明 |
| ----         | ----    | ----  | ---- |
| [precision](#precision)    | None    | --precision q8 | 模型的精度,若使用量化精度，则需指定量化苏剧 |
| [load_data_func](#load_data_func)    | load_image    | --load_data_func load_image | 模型的精度,若使用量化精度，则需指定量化苏剧 |

| 图片量化  | 默认值   | 示例     | 说明 |
| ----         | ----    | ----  | ---- |
| [image_dir](#load_image)    | None    | --image_dir image  | 图片文件夹,为None时使用内置图片                 |
| [image_color](#load_image)  | rgb     | --image_color bgr   | 训练模型所用的图片的颜色空间, rgb或bgr          |
| [image_mean](#load_image)   | 0.0     | --image_mean 0.485,0.456,0.406    | 图片的均值, img = img - mean          |
| [image_std](#load_image)    | 1.0     | --image_std 255.0,255.0,255.0   | 图片的方差, img = img / std           |
| [image_scale](#load_image)  | 1.0     | --image_scale 1/255.0,1/255.0,1/255.0 | 缩放系数   img = img * scale  |


| 模型优化         | 默认值   | 示例     | 说明 |
| ----         | ----    | ----  | ---- |
| [input_as_nhwc](#input_as_nhwc)    | None    | --input_as_nhwc true | 将模型输入的layout由nchw转为nhwc |
| [output_as_nhwc](#output_as_nhwc)    | None    | --output_as_nhwc true | 将模型输出的layout由nchw转为nhwc |
| [insert_bn](#insert_bn)    | None    | --insert_bn true | 模型做输入数据归一化，该参数依赖--image_mean image_std image_scale |
| [model_swapBR](#model_swapBR)    | None    | --model_swapBR true | rgb模型和bgr模型互转 |

| 目标检测算子         | 默认值   | 示例     | 说明 |
| ----         | ----    | ----  | ---- |
| [add_detect](#add_detect)  | false    | --add_detect true | 向网络增加目标检测大算子 |
| [detect_add_permute_node](#add_detect) | true    | --detect_add_permute_node true | 目标检测算子只支持nhwc，使用permute转换layput |
| [detect_bias](#add_detect) | yolov3的anchor    | --detect_bias 116,90,156,198,373,326,30,61,62,45,59,119,10,13,16,30,33,23 | anchor box |
| [detect_num_class](#add_detect) | 80    | --detect_num_class 80 | 目标检测的类别数 |
| [detect_conf](#add_detect) | 0.0005    | --detect_conf 0.3 | 目标检测的置信度 |
| [detect_nms](#add_detect)  | 0.45    | --detect_conf 0.45 | nms的阈值 |
| [detect_image_shape](#add_detect)  | None    | --detect_image_shape 640,640 | 目标检测图片的shape，默认根据input shape推导 |

| 调试参数         | 默认值   | 示例     | 说明 |
| ----         | ----    | ----  | ---- |
| [print_ir](#print_ir)    | false    | --print_ir true | 保存模型build的过程                 |

