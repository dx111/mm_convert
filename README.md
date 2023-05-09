# mm_convert

# 介绍
mm_convert是一个命令行工具，使用该工具可以快速的将pytorch，onnx，caffe，tensorflow模型转为magcimind模型。    
[详细文档请参考](https://dx111.github.io/mm_convert/)

# 安装
```bash
pip install mm_convert
```

# 快速开始
⚠注意：以下命令生成的模型，性能不能达到最佳，建议增加优化参数
## caffe模型转mm
```bash
mm_convert \
    --framework caffe \
    --proto resnet50.prototxt \
    --model resnet50.caffemodel \
    --output_model caffe_resnet50_model
```
## onnx模型转mm
```bash
mm_convert \
    -f onnx \
    --model densenet-12.onnx \
    --output_model onnx_densenet121_model
```
## pytorch模型转mm
```bash
mm_convert \
    -f pt \
    --model resnet50_jit.pt \
    --output_model pt_resnet50_model
```
## tensorflow pb模型转mm
```bash
mm_convert \
    -f tf \
    --model resnet50_v1.pb \
    --output_model pt_resnet50_model \
    --tf_graphdef_inputs input:0 \
    --tf_graphdef_outputs resnet_v1_50/predictions/Softmax:0
```
