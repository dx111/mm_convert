import numpy as np
import magicmind.python.runtime as mm
from mm_convert.utils import get_obj

def find_all_paths(network, graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []

    paths = []
    for node in graph[start]:
        if network.find_node_by_name(node).get_node_type() == "IConvNode":
            continue
        if node not in path:            
            newpaths = find_all_paths(network, graph, node, end, path)
            for newpath in newpaths:
                if newpath != []:
                    paths.append(newpath) 
    return paths
    
    
def find_conv_feat(network):    
    graph = {}
    for node in network.get_all_nodes_in_network():
        name = node.get_node_name()
        output_tensor = [node.get_output(i) for i in range(node.get_output_count())]
        nodes = []
        for tensor in output_tensor:
            for node in tensor.get_consumers():
                nodes.append(node.get_node_name())
        graph[name] = nodes
    
    network_output_tensor = [network.get_output(i) for i in range(network.get_output_count())]
    network_output_node_name = []
    for tensor in network_output_tensor:
        network_output_node_name.append(tensor.get_producer().get_node_name())
    convnode_set = set()
    for node in network.get_all_nodes_in_network():
        if node.get_node_type() == "IConvNode":
            name = node.get_node_name()
            for end_name in network_output_node_name:
                paths = find_all_paths(network, graph, name, end_name)
                if len(paths) != 0:
                    convnode_set.add(name)
    output_tensor = []
    for name in convnode_set:
        node = network.find_node_by_name(name)
        output_tensor.append(node.get_output(0))
    return output_tensor

def add_detect(args):
    feat_tensor = find_conv_feat(args.__network__)
    feat_tensor = sorted(feat_tensor, key = lambda x:np.prod(x.get_dimension().GetDims()), reverse=True)
    for tensor in feat_tensor:
        print("feature tensor shape: ",tensor.get_dimension())

    output_tensors = []
    for i, tensor in enumerate(feat_tensor):
        if args.detect_algo == mm.IDetectionOutputAlgo.YOLOV5:
            sigmoid_node = args.__network__.add_i_activation_node(tensor, activation_type=mm.IActivation.SIGMOID)
            tensor = sigmoid_node.get_output(0)        
        if args.detect_add_permute_node:
            dim = mm.Dims([len(args.detect_perms)])
            perms = np.array(args.detect_perms, dtype=np.int32)
            const_node = args.__network__.add_i_const_node(mm.DataType.INT32, dim, perms)
            permute_node = args.__network__.add_i_permute_node(tensor, const_node.get_output(0))
            output_tensors.append(permute_node.get_output(0))
        else:
            output_tensors.append(tensor)
    output_count = args.__network__.get_output_count()
    for i in range(output_count):
        args.__network__.unmark_output(args.__network__.get_output(0))

    bias_node = args.__network__.add_i_const_node(mm.DataType.FLOAT32, mm.Dims([len(args.detect_bias)]),
                                         np.array(args.detect_bias, dtype=np.float32))
    detect_out = args.__network__.add_i_detection_output_node(
        output_tensors, bias_node.get_output(0))
    detect_out.set_layout(mm.Layout.NONE, mm.Layout.NONE)
    detect_out.set_algo(args.detect_algo)
    detect_out.set_confidence_thresh(args.detect_conf)
    detect_out.set_nms_thresh(args.detect_nms)
    detect_out.set_scale(args.detect_scale)
    detect_out.set_num_coord(args.detect_num_coord)
    detect_out.set_num_class(args.detect_num_class)
    detect_out.set_num_entry(args.detect_num_entry)
    detect_out.set_num_anchor(args.detect_num_anchor)
    detect_out.set_num_box_limit(args.detect_box_limit)
    if not args.detect_image_shape:
        input_shape = args.__network__.get_input(0).get_dimension().GetDims()
        if args.image_size:
            size = get_obj(0, args.image_size)
        else:
            size = None
        if not size:
            if 3 == input_shape[1] or 1 == input_shape[1]:  # nchw
                size = tuple(input_shape[2:])
            if 3 == input_shape[3] or 1 == input_shape[3]:  # nhwc
                size = tuple(input_shape[1:3])
        if not size:
            print("failed to parse network input shape and image_size not define , set image_size to (416, 416)")
            size = (416, 416)
        height, width = size
    else:
        height, width = args.detect_image_shape
    detect_out.set_image_shape(width, height)
    detection_output_count = detect_out.get_output_count()
    for i in range(detection_output_count):
        args.__network__.mark_output(detect_out.get_output(i))

def model_swapBR(args):
    # todo support multi input 
    input_name = args.__network__.get_input(0).get_tensor_name()
    input_ops = []
    for node in args.__network__.get_all_nodes_in_network():
        for i in range(node.get_input_count()):
            if node.get_input(i).get_tensor_name() == input_name:
                input_ops.append(node)
    for node in input_ops:
        if node.get_node_type() == "IConvNode":
            weight_tensor = node.get_input(1)
            const_node = args.__network__.add_i_const_node(mm.DataType.INT32, mm.Dims([1]), 1)
            split_node = args.__network__.add_i_split_node(weight_tensor, const_node.get_output(0), 3)
            split_node_output = [split_node.get_output(x) for x in [2,1,0]]
            concat_node = args.__network__.add_i_concat_node(const_node.get_output(0), split_node_output)
            node.update_input(1, concat_node.get_output(0))
        else:
            raise Exception("model_swapBR only support first node is conv.")