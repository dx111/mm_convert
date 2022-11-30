mm_convert \
    -f caffe \
    --proto ../cnbox_resource/models/caffe_yolov3/yolov3_416.prototxt \
    --caffemodel ../cnbox_resource/models/caffe_yolov3/yolov3_416.caffemodel \
    --output_model caffe_yolov3_model \
    --archs mtp_322 mtp_372.41 \
    --input_shapes 1,3,416,416 \
    --input_as_nhwc true \
    --insert_bn true \
    --precision qint8_mixed_float16 \
    --calib_data_dir sample_data/coco \
    --image_color bgr \
    --image_std 1/0.00392,1/0.00392,1/0.00392 \
    --add_detect true

