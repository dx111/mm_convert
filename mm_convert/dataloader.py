import os
import cv2
import numpy as np
from glob import glob
import pickle
from typing import List, Union
import magicmind.python.runtime as mm
from magicmind.python.common.types import get_numpy_dtype_by_datatype
from mm_convert.utils import Register
from mm_convert.utils import get_obj
from mm_convert.utils import print_error_and_exit

dataload_func = Register()

@dataload_func
def load_calibrate_data(args)-> Union[List[np.ndarray], List[List[np.ndarray]]]:
    with open(args.calibrate_data_file, "rb") as f:
       calibrate_data = pickle.load(f)
    data = [calibrate_data[k] for k in calibrate_data.keys()]
    batch_size = args.batch_size
    batch_data_list = []
    input_types = []
    for i in range(args.__network__.get_input_count()):
        type = args.__network__.get_input(i).get_data_type()
        np_type = get_numpy_dtype_by_datatype(type)
        input_types.append(np_type)
    for i in range(len(data)):
        batch_data = []
        tmp_data = []
        for j in range(len(data[i])):
            tmp_data.append(data[i][j])
            if len(tmp_data) == batch_size:
                new_data = np.concatenate(tmp_data, axis=0).astype(input_types[i])
                batch_data.append(new_data)
                tmp_data = []
        batch_data_list.append(batch_data)
    return batch_data_list
    
@dataload_func
def load_image(args):
    sample_data_list = []
    for idx, data_dir in enumerate(args.image_dir):
        input_shape = args.__network__.get_input(idx).get_dimension().GetDims()
        if args.image_size:
            size = get_obj(idx, args.image_size)
        else:
            size = None
        if not size:
            if 3 == input_shape[1] or 1 == input_shape[1]:  # nchw
                size = tuple(input_shape[2:])
            if 3 == input_shape[3] or 1 == input_shape[3]:  # nhwc
                size = tuple(input_shape[1:3])
        if not size:
            print("failed to parse network input shape and image_size not define , set image_size to (224, 224)")
            size = (224, 224)
        mean = get_obj(idx, args.image_mean)
        std = get_obj(idx, args.image_std)
        scale = get_obj(idx, args.image_scale)
        image_color = get_obj(idx, args.image_color)
        img_files = glob(data_dir + "/*")
        if len(img_files) == 0:
            print_error_and_exit(f"image_dir: {data_dir} is empty")
        sample_data = []
        batch_data = []
        for file in img_files:
            img = cv2.imread(file)
            if image_color.lower() == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif image_color.lower() == "bgr":
                img = img
            elif image_color.lower() == "gray":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, size)
            if 2 == len(img.shape):
                img = np.expand_dims(img, axis=-1)
            img = img.astype("float32")
            img *= scale
            img -= mean
            img /= std
            batch_data.append(img)
            if len(batch_data) == args.batch_size:
                batch_data = np.array(batch_data)
                if tuple(input_shape[1:]) != batch_data.shape[1:]:
                    batch_data = batch_data.transpose(0, 3, 2, 1)
                if tuple(input_shape[1:]) != batch_data.shape[1:]:
                    raise BaseException("preprocess data error , input_shape: ",input_shape,", data shape: ", batch_data.shape,)
                sample_data.append(batch_data)
                batch_data = []
        sample_data_list.append(sample_data)
    return sample_data_list
