import os
import numpy as np
import cv2
import json
import logging
import argparse
from tqdm import tqdm
import magicmind.python.runtime as mm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tests.mm_infer import MMInfer

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

def yolo_postprocess(bboxes, detection_num, img_id, height, width, inv_map, input_size=416):
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
        cat_id = inv_map.get(detection_class, -1)
        result = [float(img_id), xmin, ymin, (xmax - xmin),
                  (ymax - ymin), score, float(cat_id)]
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
    model = MMInfer("caffe_yolov3_model",os.environ.get('REMOTE_IP', None))
    batch_size = 1
    coco_json_path = "../cnbox_resource/dataset/coco1000/instances_1000.json"
    image_file_path ="../cnbox_resource/dataset/coco1000/images"
        
    coco = COCO(coco_json_path)
    inv_map = load_inv_map(coco_json_path)
    detections = []
    coco_dataset = load_coco(coco, image_file_path, batch_size)
    for processed_img, extra_info in tqdm(coco_dataset, desc= "处理进度",):
        numpy_outputs = model.predict([processed_img])
        for idx in range(len(extra_info)):
            boxes = numpy_outputs[0][idx]
            detection_num = numpy_outputs[1][idx]
            img_id = extra_info[idx]["image_id"]
            height = extra_info[idx]["height"]
            width = extra_info[idx]["width"]
            results = yolo_postprocess(
                boxes, detection_num, img_id, height, width, inv_map)
            detections.extend(results)
    cocoDt = coco.loadRes(np.array(detections))
    cocoEval = COCOeval(coco, cocoDt, iouType='bbox')
    cocoEval.params.imgIds = [int(img_id) for img_id in coco.imgs.keys()]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print("mAP: ", cocoEval.stats[1])
