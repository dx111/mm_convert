---
layout: default
title: 通用参数
parent: 参数介绍
nav_order: 1
permalink: /params/common
---

### framework
原模型的框架，caffe，onnx，pytorch，tensorflow，pytorch可以使用简写pt，tensorflow可以使用简写tf
| 框架       | 参数值  |
| ----       | ---- |
| caffe      | caffe |
| pytorch    | pytorch, pt, torch |
| onnx       | onnx |
| tensorflow | tensorflow 或 tf |

example:
```bash
--framework caffe
-f onnx
-f pt
-f tf
```

### model
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

### proto 
caffe 模型的prototxt文件
```
--proto resnet50.prototxt
```

### output_model
生成mm模型的输出目录，当未指定的时候，默认生成原模型名字相同，增加.mm后缀，例如models/resnet50.onnx，会生成resnet50.onnx.mm的模型
```bash
--output_model model.mm
```

### archs
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

### input_shapes
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

### graph_shape_mutable
设置模型的输入可变
```bash
--graph_shape_mutable true
```
设置模型的输入不可变
```bash
--graph_shape_mutable true
```

