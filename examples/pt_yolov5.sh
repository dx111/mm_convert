wget -c https://github.com/dx111/models/raw/main/pytorch_yolov5/yolov5m_v5.pt \
    -p models/pytorch_yolov5

mm_convert \
    --framework pytorch \
    --model models/pytorch_yolov5/yolov5m_v5.pt\
    --output_model pytorch_yolov5m_v5_model \
    --archs mtp_322 mtp_372.41 \
    --input_shapes 1,3,640,640 \
    --input_as_nhwc true \
    --insert_bn true \
    --precision qint8_mixed_float16 \
    --image_color rgb \
    --image_scale 1/255.0,1/255.0,1/255.0 \
    --add_detect true \
    --detect_bias 10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326 \
    --detect_algo yolov5
