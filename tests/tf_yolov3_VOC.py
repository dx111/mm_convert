import os
import numpy as np
import cv2
import json
import logging
import argparse
from tqdm import tqdm
import xml.etree.ElementTree as ET
from mean_average_precision import MetricBuilder

import magicmind.python.runtime as mm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tests.mm_infer import MMInfer

name_map = {
    "background": 0,"aeroplane": 1,"bicycle": 2,"bird": 3,"boat": 4,"bottle": 5,
    "bus": 6,"car": 7,"cat": 8,"chair": 9,"cow": 10,
    "diningtable": 11,"dog": 12,"horse": 13,"motorbike": 14, "person": 15,
    "pottedplant": 16,"sheep": 17,"sofa": 18,"train": 19,"tvmonitor": 20
}

def parse_xml(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        
        name = obj.find('name').text
        difficult = int(obj.find('difficult').text)
        
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        class_id = name_map[name]
        crowd = 0
        objects.append([xmin, ymin, xmax, ymax, class_id, difficult, crowd])
    objects = np.array(objects)
    return objects


def preprocess(img, need_transpose, need_normlize, dtype):
    """
      code is based on https://github.com/ChenYingpeng/caffe-yolov3/blob/master/src/image.cpp
    """
    import math
    size = img.shape
    min_side = 416
    h, w = size[0], size[1]

    scale = float(min_side) / float(max(h, w))
    new_w, new_h = int(math.floor(float(w) * scale)
                       ), int(math.floor(float(h) * scale))
    resize_img = cv2.resize(img, (new_w, new_h), cv2.INTER_LINEAR)

    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom = (min_side-new_h)/2, (min_side-new_h)/2
        left, right = (min_side-new_w)/2 + 1, (min_side-new_w)/2
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom = (min_side-new_h)/2 + 1, (min_side-new_h)/2
        left, right = (min_side-new_w)/2, (min_side-new_w)/2
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom = (min_side-new_h)/2, (min_side-new_h)/2
        left, right = (min_side-new_w)/2, (min_side-new_w)/2
    else:
        top, bottom = (min_side-new_h)/2 + 1, (min_side-new_h)/2
        left, right = (min_side-new_w)/2 + 1, (min_side-new_w)/2

    pad_img = cv2.copyMakeBorder(resize_img, int(top),
                                 int(bottom), int(left), int(right),
                                 cv2.BORDER_CONSTANT, value=[128, 128, 128])
    image = cv2.cvtColor(pad_img, cv2.COLOR_BGR2RGB)

    if need_normlize:
        image = image.astype(np.float32)
        std = [0.00392, 0.00392, 0.00392]
        image *= std
    if need_transpose:
        image = np.transpose(image, (2, 0, 1))
    return image.astype(dtype)


def load_coco(coco, image_file_path, batch_size=1):
    image_list = []
    label_list = []
    for img_id in coco.imgs.keys():
        file_name = coco.imgs[img_id]['file_name']
        img_path = os.path.join(image_file_path, file_name)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        img = preprocess(img, need_transpose=False,
                         need_normlize=False, dtype=np.uint8)
        image_list.append(img)
        label_list.append(
            {"height": height, "width": width, "image_id": img_id})
        if len(image_list) == batch_size:
            yield np.array(image_list), label_list
            image_list = []
            label_list = []

def cal_dimension(detection, length, scale, input_dim):
    real_detection = (detection * input_dim -
                      (input_dim - length * scale) / 2) / scale
    real_detection = min(length, max(0, real_detection))
    return real_detection

def yolo_postprocess(bboxes, detection_num, height, width, input_size=416):
    scale = float(input_size) / float(max(height, width))
    if height > width:
        scale_h = 1
        scale_w = scale
    else:
        scale_h = scale
        scale_w = 1
    results = []
    for i in range(detection_num):
        detection_class = int(bboxes[i, 1]) + 1
        score = bboxes[i, 2]
        xmin = bboxes[i, 3]
        ymin = bboxes[i, 4]
        xmax = bboxes[i, 5]
        ymax = bboxes[i, 6]
        scaling_factor = min(scale_h, scale_w)
        xmin = cal_dimension(xmin, width, scaling_factor, input_size)
        ymin = cal_dimension(ymin, height, scaling_factor, input_size)
        xmax = cal_dimension(xmax, width, scaling_factor, input_size)
        ymax = cal_dimension(ymax, height, scaling_factor, input_size)
        result = [xmin, ymin, xmax, ymax, detection_class, score]
        results.append(result)
    return results

def load_inv_map(coco_json_path):
    label_map = {}
    with open(coco_json_path) as fin:
        annotations = json.load(fin)
    for cnt, cat in enumerate(annotations["categories"]):
        label_map[cat["id"]] = cnt + 1
    inv_map = {v: k for k, v in label_map.items()}
    return inv_map

if __name__ == "__main__":
    model = MMInfer("tf_yolov3_model")
    batch_size = 1

    val_file = "/data/datasets-common/VOC/test/VOCdevkit/VOC2007/Annotations"
    file_list = "/data/datasets-common/VOC/test/VOCdevkit/VOC2007/ImageSets/Main/test.txt"
    image_path = "/data/datasets-common/VOC/test/VOCdevkit/VOC2007/JPEGImages"
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes= len(name_map.keys()))

    with open(file_list, 'r') as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
        
    for line in tqdm(lines, unit='img', desc='处理进度'):
        jpeg_file = os.path.join(image_path, line + '.jpg')
        img = cv2.imread(jpeg_file)
        height,width = img.shape[:2]
        image_list = []
        processed_img = preprocess(img, need_transpose=False,
                         need_normlize=False, dtype=np.uint8)
        image_list.append(processed_img)
        processed_img = np.array(image_list)
        
        numpy_outputs = model.predict([processed_img])

        # for idx in range(len(extra_info)):
        boxes = numpy_outputs[0][0]
        detection_num = numpy_outputs[1][0]
        # img_id = extra_info[idx]["image_id"]
        # height = extra_info[idx]["height"]
        # width = extra_info[idx]["width"]
        results = yolo_postprocess(
            boxes, detection_num, height, width)
        results = np.array(results)
        gt = parse_xml(os.path.join(val_file, line + '.xml'))
        metric_fn.add(results, gt)
    map = metric_fn.value(iou_thresholds=0.45, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']
    print("map: ", map)