---
layout: default
title: 目标检测
parent: 参数介绍
nav_order: 4
---

### add_detect
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

