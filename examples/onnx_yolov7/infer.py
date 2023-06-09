from typing import List
import numpy as np
import magicmind.python.runtime as mm
import cv2
import math
from mm_convert.tools.mm_infer import MMInfer

model_file = "yolov7.mm"
img_size = [640, 640]
image_files = [
    "./data/zidane.jpg",
    "./data/bus.jpg"
]

def preprocess(img, dst_shape = [640, 640], stride = 32):
    src_h, src_w = img.shape[0], img.shape[1]
    dst_h, dst_w = dst_shape
    ratio = min(dst_h / src_h, dst_w / src_w)
    unpad_h, unpad_w = int(math.floor(src_h * ratio)), int(math.floor(src_w * ratio))
    if ratio != 1:
        interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (unpad_w, unpad_h), interp)

    pad_t = 0
    pad_b = dst_h - unpad_h
    pad_l = 0
    pad_r = dst_w - unpad_w
    img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0,0,0))
    return img, ratio

model = MMInfer(model_file)
img_list = []
show_img_list = []
ratio_list = []
for img_file in image_files:
    img = cv2.imread(img_file)
    show_img_list.append(cv2.imread(img_file))
    img, ratio = preprocess(img)
    img_list.append(img)
    ratio_list.append(ratio)

batch_image = np.stack(img_list, axis=0)
outs = model.predict(batch_image)

for b_idx, box_num in enumerate(outs[1]):
    scale =  ratio_list[b_idx]
    show_img = show_img_list[b_idx]
    boxes = outs[0][b_idx]
    for idx in range(box_num):
        class_id = int(boxes[idx][1])
        score    = boxes[idx][2]
        if score < 0.3:
            continue
        left     = boxes[idx][3]
        top      = boxes[idx][4]
        right    = boxes[idx][5]
        bottom   = boxes[idx][6]
        
        left *= 1.0/scale
        top *= 1.0/scale
        right *= 1.0/scale
        bottom *= 1.0/scale
        
        cv2.rectangle(show_img,(int(left), int(top)), (int(right), int(bottom)), (255,255,0), 2)
    cv2.imwrite(f"python_result_{b_idx}.jpg", show_img)
        