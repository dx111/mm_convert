import os
import argparse
import magicmind.python.runtime as mm

HERE = os.path.split(os.path.abspath(__file__))[0]

toolchain_path = "/tmp/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/"

bool_flags = ["1","t","T","true","True","0","f","F","false","False"]

def str_to_int_list(string:str):
    return [int(x) for x in string.split(",")]

def str_to_flaot_list(string:str):
    return [float(eval(x)) for x in string.split(",")]

def str_to_int_tuple(string:str):
    value = string.split(",")
    return (int(value[0]), int(value[1]))

def str_to_bool(string:str):
    if string in ["0","f","F","false","False"]:
        return False
    else:
        return True

def str_to_mm_dtype(string: str):
    if string.lower() == "float32":
        return mm.DataType.FLOAT32
    elif string.lower() == "float64":
        return mm.DataType.FLOAT64
    elif string.lower() == "int32":
        return mm.DataType.INT32
    elif string.lower() == "int64":
        return mm.DataType.INT64
    else:
        raise BaseException("unsuporrt")

def str_to_framework(string:str):
    if string.lower() == "caffe":
        return "caffe"
    elif string.lower() == "tf" or string.lower() == "tensorflow":
        return "tensorflow"
    elif string.lower() == "pt" or string.lower() == "pytorch" or string.lower() == "torch":
        return "pytorch"
    elif string.lower() == "onnx":
        return "onnx"
    else:
        raise BaseException(f"invalid framework option {string}")
    
def str_to_arch(string: str):
    value = string.split(":")
    if len(value) == 2:
        return {value[0]: [int(x) for x in value[1].split(",")]}
    elif len(value) == 1:
        return value[0]
    else:
        raise BaseException("unsupport")

str_to_precision_dict = {
    "qint8_mixed_float16": "qint8_mixed_float16",
    "q8_f16":"qint8_mixed_float16",
    "q8":"qint8_mixed_float16",

    "qint8_mixed_float32":"qint8_mixed_float32",
    "q8_f32":"qint8_mixed_float32",

    "qint16_mixed_float16":"qint16_mixed_float16",
    "q16_f16":"qint16_mixed_float16",
    "q16":"qint16_mixed_float16",

    "qint16_mixed_float32":"qint16_mixed_float32",
    "q16_f32":"qint16_mixed_float32",

    "force_float16":"force_float16",
    "f16":"force_float16",

    "force_float32":"force_float32",
    "f32":"force_float32"
}

def str_to_precision(string: str):
    if string in str_to_precision_dict.keys():
        return str_to_precision_dict[string]
    else:
        raise BaseException(f"failed to paser precision, value must in {str_to_precision_dict.keys()}")
    
# add detect
def str_to_detect_algo(string: str):
    if string.lower() == "yolov2":
        return mm.IDetectionOutputAlgo.YOLOV2
    elif string.lower() == "yolov3":
        return mm.IDetectionOutputAlgo.YOLOV3
    elif string.lower() == "yolov4":
        return mm.IDetectionOutputAlgo.YOLOV4
    elif string.lower() == "yolov5":
        return mm.IDetectionOutputAlgo.YOLOV5
    elif string.lower() == "fasterrcnn":
        return mm.IDetectionOutputAlgo.FASTERRCNN
    elif string.lower() == "ssd":
        return mm.IDetectionOutputAlgo.SSD
    elif string.lower() == "refinedet":
        return mm.IDetectionOutputAlgo.REFINEDET
    else:
        raise BaseException("unsupport IDetectionOutputAlgo", string)    

parser = argparse.ArgumentParser()

# framework
parser.add_argument("-f","--framework",type=str, required = True , metavar="caffe,pytorch,onnx,tensorflow,pytorch")
parser.add_argument("-m","--model", type=str)
parser.add_argument("-o","--output_model",type=str)
parser.add_argument("--input_shapes",type=str_to_int_list, nargs="+")

# caffe
caffe_param_group = parser.add_argument_group("param for caffe framework")
caffe_param_group.add_argument("--proto", type=str)

# pytorch
pt_param_group = parser.add_argument_group("param for pytorch framework")
pt_param_group.add_argument("--pytorch_input_dtypes", type= str_to_mm_dtype ,nargs = "+" , default= [mm.DataType.FLOAT32])

# tensorflow
tf_param_group = parser.add_argument_group("param for tf framework")
tf_param_group.add_argument("--tf_infer_shape",type=str_to_bool , default= False)
tf_param_group.add_argument("--tf_model_type", type=str,choices=["tf-graphdef-file", "tf-savedmodel-v1-dir"], default="tf-graphdef-file")
# pb
tf_param_group.add_argument("--tf_graphdef_inputs", type=str, nargs="+")
tf_param_group.add_argument("--tf_graphdef_outputs", type=str, nargs="+")
# saved_model
tf_param_group.add_argument("--tf_savedmodel_tags", type=str,default= ["serve"], nargs="+")
tf_param_group.add_argument("--tf_savedmodel_exported_names",default =["serving_default"], type=str, nargs="+")
tf_param_group.add_argument("--tf_savedmodel_main_func",default ="serving_default", type=str)

# build_config
build_config_group = parser.add_argument_group("param for build_config")
build_config_group.add_argument("--costom_build_config", type= str, default= None)
build_config_group.add_argument("--archs", nargs= "+", type=str_to_arch)
build_config_group.add_argument("--graph_shape_mutable", type=str_to_bool)
build_config_group.add_argument("--input_as_nhwc", type = str_to_bool, default = [False], nargs= "+", metavar="false")
build_config_group.add_argument("--output_as_nhwc", type = str_to_bool, default = [False], nargs= "+", metavar="false")
build_config_group.add_argument("--insert_bn", type=str_to_bool, default = [False], nargs= "+", metavar="false")
build_config_group.add_argument("--precision", type = str_to_precision, metavar="qint8_mixed_float16")
build_config_group.add_argument("--type64to32_conversion", type=str_to_bool, metavar= "true")
build_config_group.add_argument("--conv_scale_fold", type=str_to_bool, metavar= "true")
build_config_group.add_argument("--print_ir", type=str_to_bool)
build_config_group.add_argument("--cross_compile_toolchain_path", type=str, default = toolchain_path, metavar = toolchain_path)
build_config_group.add_argument("--compute_determinism", type=str_to_bool, default= False)

# calibrate
calibrate_group = parser.add_argument_group("param for preprocess and calibrate")
calibrate_group.add_argument("--remote_ip", type=str, metavar="None")
calibrate_group.add_argument("--load_data_func", type=str, default = "load_image")
calibrate_group.add_argument("--batch_size", type= int, default=1)

# load_image
calibrate_group.add_argument("--image_dir", type=str, nargs="+", default= [f"{HERE}/sample_data/"])
calibrate_group.add_argument("--image_size", type = str_to_int_tuple, nargs="+")
calibrate_group.add_argument("--image_color", type=str, nargs="+", default= ["RGB"], metavar="rgb")
calibrate_group.add_argument("--image_mean", type=str_to_flaot_list, nargs="+", default= [[0.0,0.0,0.0]], metavar="0.0,0.0,0.0")
calibrate_group.add_argument("--image_std", type=str_to_flaot_list, nargs="+", default=[[1.0,1.0,1.0]], metavar="1.0,1.0,1.0")
calibrate_group.add_argument("--image_scale", type=str_to_flaot_list, nargs= "+", default=[[1.0,1.0,1.0]], metavar="1.0,1.0,1.0")

# load_squad
calibrate_group.add_argument("--squad_max_seq_length", type=int, default= 128)
calibrate_group.add_argument("--squad_doc_stride", type=int, default = 128 )
calibrate_group.add_argument("--squad_max_query_length", type=int, default=64)

# calibrate_data
calibrate_group.add_argument("--calibrate_data_file", type=str, default="calibrate_data")

# detect_param 
detect_param = parser.add_argument_group("param for add detectionout node param")
detect_param.add_argument("--add_detect", type= str_to_bool, default= False)
detect_param.add_argument("--detect_add_permute_node", type= str_to_bool, default= True)
detect_param.add_argument("--detect_perms", type= str_to_int_list, default=[0, 2, 3, 1], metavar= "0,2,3,1")
detect_param.add_argument("--detect_bias", type= str_to_int_list, default=[116, 90, 156, 198, 373, 326, 30, 61, 62, 45, 59, 119, 10, 13, 16, 30, 33, 23], metavar="116,90,156,198,373,326,30,61,62,45,59,119,10,13,16,30,33,23")
detect_param.add_argument("--detect_algo", type= str_to_detect_algo, default= mm.IDetectionOutputAlgo.YOLOV3)
detect_param.add_argument("--detect_conf", type= float, default=0.0005, metavar="0.0005")
detect_param.add_argument("--detect_nms", type= float, default=0.45, metavar="0.45")
detect_param.add_argument("--detect_scale", type= float, default=1.0, metavar="1.0")
detect_param.add_argument("--detect_num_coord", type= int, default=4, metavar="4")
detect_param.add_argument("--detect_num_class", type= int, default=80, metavar="80")
detect_param.add_argument("--detect_num_entry", type= int, default=5, metavar="5")
detect_param.add_argument("--detect_num_anchor", type= int, default=3, metavar="3")
detect_param.add_argument("--detect_box_limit", type= int, default=2048, metavar="2048")
detect_param.add_argument("--detect_image_shape", type= str_to_int_list, default=[-1, -1], metavar="-1,-1")

# swap model rgb2bgr or bgr2rgb
swap_BR_param = parser.add_argument_group("param for swap model rgb2bgr or bgr2rgb ")
swap_BR_param.add_argument("--model_swapBR", type=str_to_bool, default=False)
