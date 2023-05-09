---
layout: default
title: 快速开始
nav_order: 2
description: "快速开始"
permalink: /overview
---


## 快速开始
{: .warning }
以下命令生成的模型，性能不能达到最佳，建议增加优化参数

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
更多示例请参开exmples目录下的脚本
