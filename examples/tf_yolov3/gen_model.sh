wget -nc -c https://github.com/dx111/models/raw/main/tf_yolov3/yolov3.pb

mm_convert \
    --framework tensorflow \
    --model yolov3.pb \
    --output_model tf_yolov3_model \
    --archs mtp_372.41 \
    --tf_graphdef_inputs  input/input_data \
    --tf_graphdef_outputs conv_lbbox/Conv2D:0 conv_mbbox/Conv2D:0 conv_sbbox/Conv2D:0 \
    --input_shapes 1,416,416,3 \
    --insert_bn true \
    --precision qint8_mixed_float16 \
    --image_color rgb \
    --image_std 255.0 255.0 255.0 \
    --add_detect true \
    --detect_add_permute_node false 
