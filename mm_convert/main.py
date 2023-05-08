import os
import numpy as np
import json
import magicmind.python.runtime as mm
import magicmind.python.runtime.parser
from mm_convert.utils import Calibrator
from mm_convert.utils import print_error_and_exit
from mm_convert.options import parser
from mm_convert.adapter_model import add_detect
from mm_convert.adapter_model import model_swapBR
from mm_convert.dataloader import dataload_func

def parse_network(args):
    network = mm.Network()
    if args.framework == "caffe":
        caffe_parser = mm.parser.Parser(mm.ModelKind.kCaffe)
        assert caffe_parser.parse(network, args.model, args.proto).ok()
    elif args.framework == "onnx":
        onnx_parser = mm.parser.Parser(mm.ModelKind.kOnnx)
        assert onnx_parser.parse(network, args.model).ok()
    elif args.framework == "pytorch":
        pt_parser = mm.parser.Parser(mm.ModelKind.kPytorch)
        pt_parser.set_model_param("pytorch-input-dtypes", args.pytorch_input_dtypes)
        assert pt_parser.parse(network, args.model).ok()
    elif args.framework == "tensorflow":
        tf_parser = mm.parser.Parser(mm.ModelKind.kTensorflow)
        tf_parser.set_model_param("tf-infer-shape", args.tf_infer_shape)
        tf_parser.set_model_param("tf-model-type", args.tf_model_type)
        if args.tf_model_type == "tf-graphdef-file":
            tf_parser.set_model_param("tf-graphdef-inputs", args.tf_graphdef_inputs)
            tf_parser.set_model_param("tf-graphdef-outputs", args.tf_graphdef_outputs)
        elif args.tf_model_type == "tf-savedmodel-v1-dir":
            tf_parser.set_model_param("tf-savedmodel-tags", args.tf_savedmodel_tags)
            tf_parser.set_model_param("tf-savedmodel-exported-names", args.tf_savedmodel_exported_names)
            tf_parser.set_model_param("tf-savedmodel-main-func", args.tf_savedmodel_main_func)
        else:
            raise BaseException("unsupport tensorflow model type")
        assert tf_parser.parse(network, args.model).ok()
    setattr(args, "__network__", network)

def get_obj(idx, objs):
    if idx >= len(objs):
        obj = objs[0]
    else:
        obj = objs[idx]
    return obj

def parse_build_config(args):
    if args.costom_build_config:
        with open(args.costom_build_config, "r") as f:
            config = json.loads(f.read())
    else:
        config = {}
        if args.archs is not None:
            config["archs"] = args.archs
        if args.graph_shape_mutable is not None:
            config["graph_shape_mutable"] = args.graph_shape_mutable
        if args.type64to32_conversion is not None:
            config["opt_config"] = {"type64to32_conversion": args.type64to32_conversion}
        if args.conv_scale_fold is not None:
            config["opt_config"] = {"conv_scale_fold": args.conv_scale_fold}
        if args.precision is not None:
            config["precision_config"] = {"precision_mode": args.precision} 
        if args.print_ir:
            config["debug_config"] = {"print_ir": {"print_level": 1}}
        if "322" in str(args.archs) or args.archs is None:
            config["cross_compile_toolchain_path"] = args.cross_compile_toolchain_path
        for idx, flag in enumerate(args.input_as_nhwc):
            if flag:
                if "convert_input_layout" not in config.keys():
                    config["convert_input_layout"] = dict()
                config["convert_input_layout"][f"{idx}"] = {"src": "NCHW", "dst": "NHWC"}

        for idx, flag in enumerate(args.output_as_nhwc):
            if flag:
                if "convert_output_layout" not in config.keys():
                    config["convert_output_layout"] = dict()
                config["convert_output_layout"][f"{idx}"] = {"src": "NCHW", "dst": "NHWC"}  

        for idx, flag in enumerate(args.insert_bn):
            if flag:
                if "insert_bn_before_firstnode" not in config.keys():
                    config["insert_bn_before_firstnode"] = dict()
                mean = get_obj(idx, args.image_mean)
                std = get_obj(idx, args.image_std)
                scale = get_obj(idx, args.image_scale)
                _mean = np.array(mean) / np.array(scale)
                _var = (np.array(std) / np.array(scale)) * (np.array(std) / np.array(scale))
                config["insert_bn_before_firstnode"][f"{idx}"] =  {"mean": _mean.tolist(), "var": _var.tolist()}
        if args.compute_determinism:
            config["compute_determinism"] = True
    print("=========== build config ===================")
    print(config)
    print("============================================")
    return config

def args_check(args):
    if args.framework == "caffe":
        if args.model is None:
            print_error_and_exit("error: the following arguments are required: --model xxxx.caffemodel") 
        if args.proto is None:
            print_error_and_exit("error: the following arguments are required: --proto xxxx.prototxt")
        if not os.path.exists(args.model):
            print_error_and_exit(f"invalid model path, {args.model} not exists.")
        if not os.path.exists(args.proto):
            print_error_and_exit(f"invalid model path, {args.proto} not exists.")
    if args.framework == "onnx":
        if args.model is None:
            print_error_and_exit("error: the following arguments are required: --model xxxx.onnx")
        if not os.path.exists(args.model):
            print_error_and_exit(f"invalid model path, {args.model} not exists.")
    if args.framework == "pytorch":
        if args.model is None:
            print_error_and_exit("error: the following arguments are required: --model xxxx.pt")
        if not os.path.exists(args.model):
            print_error_and_exit(f"invalid model path, {args.model} not exists.")

    if args.framework == "tensorflow":
        if args.tf_model_type == "tf-graphdef-file":
            if args.tf_graphdef_inputs is None:
                print_error_and_exit("error: the following arguments are required: --tf_graphdef_inputs input_name1 input_name2")
            if args.tf_graphdef_outputs is  None:
                print_error_and_exit("error: the following arguments are required: --tf_graphdef_outputs output_name1 output_name2")
        if not os.path.exists(args.model):
            print_error_and_exit(f"invalid model path, {args.model} not exists.")
    # set default output model name
    if args.output_model is None:
        setattr(args, "output_model", os.path.split(args.model)[-1] + ".mm")
    # set batch size
    if args.batch_size is None and args.input_shapes:
        setattr(args, "batch_size", args.input_shapes[0][0])
    
def main():
    twis_info = dict()
    args = parser.parse_args()
    args_check(args)
    parse_network(args)
    if args.input_shapes:
        twis_info["input_shapes"] = args.input_shapes
        for i in range(len(args.input_shapes)):
            shape = mm.Dims(args.input_shapes[i])
            args.__network__.get_input(i).set_dimension(shape)
        if args.graph_shape_mutable is None:
            setattr(args, "graph_shape_mutable", False)
    if args.model_swapBR:
        model_swapBR(args)
    if args.add_detect:
        add_detect(args)
    build_config = mm.BuilderConfig()
    config = parse_build_config(args)
    twis_info["build_config"] = config
    json_data = json.dumps(config)
    assert build_config.parse_from_string(json_data).ok()
    if args.precision and args.precision not in ["force_float16", "force_float32"]:
        calib_data = dataload_func[args.load_data_func](args)
        calibrator = Calibrator(calib_data, mm.QuantizationAlgorithm.LINEAR_ALGORITHM)
        if args.remote_ip:
            remote_config = mm.RemoteConfig()
            remote_config.address = args.remote_ip + ":9009"
            calibrator.set_remote(remote_config)
        assert calibrator.calibrate(args.__network__, build_config).ok()
    builder = mm.Builder()
    model = builder.build_model("mm_model", args.__network__, build_config)
    assert model != None
    model.serialize_to_file(args.output_model)
    with open(f'{args.output_model}_twins.json', 'w') as f:
        json.dump(twis_info, f)
    print("Generate model done, model save to %s" % args.output_model)

if __name__ == "__main__":
    main()
