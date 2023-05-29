import os
import cv2
import numpy as np
from tqdm import tqdm
from tests.mm_infer import MMInfer

def imagenet_dataset(imagenet_path = "/data/pytorch/datasets/imagenet", 
                     val_txt = "ILSVRC2012_val.txt", 
                     count=-1):
    with open(val_txt, "r") as f:
        lines = f.readlines()
    current_count = 0
    for line in lines:
        image_name, label = line.split(" ")
        image_path = os.path.join(imagenet_path, image_name)
        img = cv2.imread(image_path)
        yield img, label.strip()
        current_count += 1
        if current_count >= count and count != -1:
            break

def preprocess_resnet_no_scale(image,
                               size = (256, 256), 
                               crop = (224, 224), 
                               transpose = False,
                               color = "rgb"):
    h, w = image.shape[:2]
    new_h = size[0]
    new_w = size[1]
    if (w < h):
        new_h = int(size[0] / w * h)
    else:
        new_w = int(size[1] / h * w)
    if color == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (new_w, new_h))
    h, w = image.shape[:2]
    oh, ow = crop
    x = int(round((w - ow) / 2.))
    y = int(round((h - oh) / 2.))
    image = image[y : y + oh, x : x + ow]
    if transpose:
        image = np.transpose(image, (2, 0, 1))
    return image


if __name__ == "__main__":
    model = MMInfer("tf_resnet50_model")
    dataset = imagenet_dataset()
    top1_count = 0
    top5_count = 0
    total_count = 0
    for img, label in tqdm(dataset, unit="img", desc= "处理进度"):
        data = preprocess_resnet_no_scale(img, color="rgb")
        data = np.expand_dims(data, 0)
        outputs = model.predict([data])
        
        index = outputs[0][0].argsort()[::-1]
        if int(label) == index[0]:
            top1_count += 1
        if int(label) in index[:5]:
            top5_count += 1
        total_count += 1
    
    top1 = float(top1_count) / float(total_count)
    top5 = float(top5_count) / float(total_count)
    print("top1 accuracy: %f"%top1)
    print("top5 accuracy: %f"%top5)