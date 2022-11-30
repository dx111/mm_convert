import os
import numpy as np
import cv2
import json
from tqdm import tqdm
import magicmind.python.runtime as mm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tests.mm_infer import MMInfer
import math


def preprocess(img, dst_shape = [640, 640], stride = 32):
    src_h, src_w = img.shape[0], img.shape[1]
    dst_h, dst_w = dst_shape
    ratio = min(dst_h / src_h, dst_w / src_w)
    unpad_h, unpad_w = int(math.floor(src_h * ratio)), int(math.floor(src_w * ratio))
    if ratio != 1:
        interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (unpad_w, unpad_h), interp)
    # padding
    pad_t = int(math.floor((dst_h - unpad_h) / 2))
    pad_b = dst_h - unpad_h - pad_t
    pad_l = int(math.floor((dst_w - unpad_w) / 2))
    pad_r = dst_w - unpad_w - pad_l
    img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(114,114,114))
    return img, ratio

def load_label_name(label_name):
    names_list = []
    with open(label_name, "r") as f:
        lines = f.readlines()
    for line in lines:
        names_list.append(line.replace("\n", ""))
    return names_list

def post_process(outs, width, height, ratio, eval = False, img_size = [640, 640],conf_thresh= 0.5, name_list = None):
    boxes = outs[0][0]
    detection_num = outs[1][0]
    scale_w = ratio * width 
    scale_h = ratio * height
    results = []
    for k in range(detection_num):
        class_id = int(boxes[k][1])
        score = float(boxes[k][2])
        left = max(0, min(boxes[k][3], img_size[1]))
        top = max(0, min(boxes[k][4], img_size[0]))
        right = max(0, min(boxes[k][5], img_size[1]))
        bottom = max(0, min(boxes[k][6], img_size[0]))

        left = (left - (img_size[1] - scale_w) / 2)  
        right = (right - (img_size[1] - scale_w) / 2) 
        top = (top - (img_size[0] - scale_h) / 2) 
        bottom = (bottom - (img_size[0] - scale_h) / 2)
        if eval:
            name = class_id
        else:
            name = name_list[class_id]
            left = int(max(0, left))
            right = int(max(0, right))
            top = int(max(0, top))
            bottom = int(max(0, bottom))
            if score < conf_thresh:
                continue
        results.append([name, score, left, top, right, bottom])
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
    coco_json = "../cnbox_resource/dataset/coco1000/instances_1000.json"
    image_path = "../cnbox_resource/dataset/coco1000/images"

    coco = COCO(coco_json)
    inv_map = load_inv_map(coco_json)
    model = MMInfer("pytorch_yolov5m_v5_model",os.environ.get('REMOTE_IP', None))
    detections = []
    for img_id in tqdm(coco.imgs.keys(), unit = "img", desc = "处理进度"):
        file_name = coco.imgs[img_id]['file_name']
        img_path = os.path.join(image_path, file_name)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        img, ratio = preprocess(img)
        img = np.expand_dims(img,0)
        outs = model.predict([img])
        results = post_process(outs, width, height, ratio, True)
        for result in results:
            class_idx, score, left, top, right, bottom = result
            cat_id = inv_map.get(class_idx + 1, -1)
            anns = coco.anns.get(img_id)
            detect = [img_id, left, top, (right - left),(bottom - top), score, cat_id]
            detect = [float(x) for x in detect]
    
            detections.append(detect)
    cocoDt = coco.loadRes(np.array(detections))
    cocoEval = COCOeval(coco, cocoDt, iouType='bbox')
    cocoEval.params.imgIds = [int(img_id) for img_id in coco.imgs.keys()]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print("mAP: ", cocoEval.stats[1])
