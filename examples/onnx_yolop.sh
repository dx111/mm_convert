wget -c https://github.com/dx111/models/raw/main/onnx_yolop/yolop-640-640.onnx \
    -P modelsonnx_yolop

mm_convert \
    --framework onnx \
    --model models/onnx_yolop/yolop-640-640.onnx \
    --output_model onnx_resnet50_model \
    --archs mtp_322 mtp_372.41 \
    --input_shapes 1,3,640,640 \
    --input_as_nhwc true \
    --insert_bn true \
    --precision qint8_mixed_float16 \
    --image_color rgb \
    --image_mean 0.485,0.456,0.406 \
    --image_std 0.229,0.224,0.225 \
    --image_scale 1/255.0,1/255.0,1/255.0
