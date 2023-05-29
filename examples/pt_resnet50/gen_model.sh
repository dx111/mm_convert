wget -nc -c https://github.com/dx111/models/raw/main/pytorch_resnet50/py3.7.9_torch1.6.0_resnet50.pt

mm_convert \
    --framework pytorch \
    --model py3.7.9_torch1.6.0_resnet50.pt \
    --output_model pt_resnet50_model \
    --archs mtp_372.41 \
    --input_shapes 1,3,224,224 \
    --input_as_nhwc true \
    --insert_bn true \
    --precision qint8_mixed_float16 \
    --image_color rgb \
    --image_mean 0.485,0.456,0.406 \
    --image_std 0.229,0.224,0.225 \
    --image_scale 1/255.0,1/255.0,1/255.0
