---
layout: default
title: YOLOV7
parent: 参考示例
nav_order: 5
permalink: /examples/yolov7
---

# yolov7

[查看代码](https://github.com/dx111/mm_convert/tree/main/examples/onnx_yolov7){: .btn .btn-blue }

1 使用官方仓库的的python程序将yolov7转为onnx模型        
```bash
python export.py --weights yolov7-tiny.pt --grid --end2end --simplify \
        --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
```
此处需要去掉--simplify参数，因为此参数会固定输入的形状，无法生成多batch模型。

2 执行转换     
请根据实际情况修改参数，此处生成的是2batch的模型     
```
mm_convert \
    -f onnx \
    --model yolov7.onnx \
    --output_model yolov7.mm \
    --archs mtp_372.41 \
    --input_shapes 2,3,640,640 \
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
此处开启了很多优化，因此参数较多:
1. --archs mtp_372.41    指定使用370设备
2. --input_shapes 2,3,640,640    执行输入的shape是2,3,640,640
3. --input_as_nhwc true   将输入从nchw转为nhwc
4. --insert_bn true 首层做归一化
5. --precision qint8_mixed_float16 设置精度
6. --image_color rgb 设置输入图片为rgb格式 【量化需要】
7. --image_scale 1/255.0,1/255.0,1/255.0 设置输入的scale【量化需要】【insert_bn需要】
8. --add_detect true 添加目标检测大算子
9. --detect_bias 12,16,19,36,40,28,36,75,76,55,72,146,142,110,192,243,459,401 设置anchor box
10. --detect_image_shape 640,640 设置检测图的尺寸
11. --detect_algo yolov5 设置检测算法为yolov5

请根据自身情况调整参数，可能会用到的参数： 
1. --detect_conf 0.3 目标检测的阈值
2. --detect_nms 0.45 目标检测iou阈值
3. --detect_num_class 80 目标检测的类别数

## python 推理
下载仓库中的infer.py文件和data目录下的两张图片，执行推理
```
python infer.py
```
会在当前目录下，保存推理python推理的结果
![python推理结果1](python_result_0.jpg)
![python推理结果2](python_result_1.jpg)

## cpp 推理
下载仓库中的infer.cpp文件, build.sh文件和data目录下的两张图片，编译代码
```
./build.sh
```
执行推理
```
./infer.bin
```
会在当前目录下，保存推理cpp推理的结果
![cpp推理结果1](cpp_res_0.jpg)
![cpp推理结果2](cpp_res_1.jpg)

## 注意事项：
1. 该脚本使用了yolov5的目标检测大算子替换原生的检测层，如果不需要，设置--add_detect false
2. 默认的detect_conf是0.3，detect_nms是0.45，类别数为80，请根据实际情况修改
3. 如需测试MaP，将detect_conf设为0.0005

## 常见问题：
1. channel不匹配：      
默认的类别数是80，255(channel)=(80(类别数)+1(分数)+4(坐标))*3(三个检测头)，设置的类别数和特征图的channel数不匹配，会报此错误
