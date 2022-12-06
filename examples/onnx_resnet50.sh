wget -c https://github.com/dx111/models/raw/main/onnx_resnet/resnet50-v1-7.onnx \
    -P models/onnx_resnet

mm_convert \
    -f onnx \
    -m models/onnx_resnet/resnet50-v1-7.onnx \
    -o onnx_resnet50_model \
    --archs mtp_372.41 \
    --input_shapes 1,3,224,224 \
    --input_as_nhwc true \
    --insert_bn true \
    --precision q8 \
    --image_color rgb \
    --image_mean 0.485,0.456,0.406 \
    --image_std 0.229,0.224,0.225 \
    --image_scale 1/255.0,1/255.0,1/255.0

# python tests/onnx_resnet50.py
