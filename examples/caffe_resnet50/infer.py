import numpy as np
import cv2
from mm_convert.tools.mm_infer import MMInfer

size = [256,256]
crop = [224,224]

with open("data/names.txt", "r") as f:
    lines = f.readlines()
    names = [''.join(s.split(' ', 1)[1:]).rstrip('\n') for s in lines]

model = MMInfer("caffe_resnet50_model")

img = cv2.imread("data/ILSVRC2012_val_00000001.JPEG")
h, w = img.shape[:2]
new_h = size[0]
new_w = size[1]
if (w < h):
    new_h = int(size[0] / w * h)
else:
    new_w = int(size[1] / h * w)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (new_w, new_h))
h, w = img.shape[:2]
oh, ow = crop
x = int(round((w - ow) / 2.))
y = int(round((h - oh) / 2.))
img = img[y : y + oh, x : x + ow]

outputs = model.predict(np.expand_dims(img, 0))
index = outputs[0][0].argsort()[::-1]

for i in range(5):
    print(f"top {i}: {names[index[i]]}, score: {outputs[0][0][index[i]]}")
