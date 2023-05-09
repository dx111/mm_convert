---
layout: default
title: YOLOV7
parent: 参考示例
nav_order: 5
permalink: /examples/yolov7
---

# yolov7

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
