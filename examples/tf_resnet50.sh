mm_convert \
    --framework tensorflow \
    --model ../cnbox_resource/models/tf_resnet50v1/resnet50_v1.pb \
    --output_model tf_resnet50_model \
    --archs mtp_372.41 \
    --tf_graphdef_inputs input:0 \
    --tf_graphdef_outputs resnet_v1_50/predictions/Softmax:0 \
    --input_shapes 1,224,224,3 \
    --input_as_nhwc false \
    --insert_bn true \
    --precision qint8_mixed_float16 \
    --image_dir sample_data/imagenet \
    --image_color rgb \
    --image_mean 123.68,116.78,103.94

# /product/datasets/cnbox_resource/models/tf_resnet50v1

# python tests/onnx_resnet50.py
