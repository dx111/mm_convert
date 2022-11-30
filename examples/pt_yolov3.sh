mm_convert \
    --framework pytorch \
    --model models/yolov3_traced.pt \
    --output_model pytorch_yolov3_model \
    --archs mtp_372.41 \
    --input_shapes 16,3,416,416 \
    --input_as_nhwc true \
    --insert_bn true \
    --precision qint8_mixed_float16 \
    --image_dir sample_data/voc \
    --image_color rgb \
    --image_std 255.0 255.0 255.0 \
    --add_detect  true \
    --detect_bias 116,90,156,198,373,326,30,61,62,45,59,119,10,13,16,30,33,23 \
    --detect_algo yolov3 