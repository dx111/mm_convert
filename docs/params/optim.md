---
layout: default
title: 模型优化
parent: 参数介绍
nav_order: 3
permalink: /params/optim
---


### input_as_nhwc
图片加载后一般的数据格式是(h,w,c)，增加batch后是(n,h,w,c),对于网络的输入是(n,c,h,w),图片需要transpose(n,h,w,c)->(n,c,h,w)后，才能进行推理，通过设置参数input_as_nhwc，将网络的输入转变为nhwc后，可免去图片的transpose      

将输入1从nchw改成nhwc
```bash
--input_as_nhwc true \
```

输入1不变，将输入2从nchw改成nhwc
```bash
--input_as_nhwc false true \
```

### output_as_nhwc
将输出1不变，将输出2从nchw改成nhwc
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

### model_swapBR
对于一些已经训练好的模型，训练时采用的bgr或者rgb的数据，推理时图片需要对图片，进行rbg2bgr或者bgr2rgb的转换，此转换浪费了时间，可对模型进行更改，交换权值中的B,R通道，使其接收另一种颜色空间的图片
```bash
--model_swapBR true
```
