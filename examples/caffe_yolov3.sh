wget -c https://github.com/dx111/models/raw/main/caffe_yolov3/yolov3_416.caffemodel \
    -P models/caffe_yolov3

wget -c https://github.com/dx111/models/raw/main/caffe_yolov3/yolov3_416.prototxt \
    -P models/caffe_yolov3

mm_convert \
    -f caffe \
    --proto models/caffe_yolov3/yolov3_416.prototxt \
    --caffemodel models/caffe_yolov3/yolov3_416.caffemodel \
    --output_model caffe_yolov3_model \
    --archs mtp_372.41 \
    --input_shapes 1,3,416,416 \
    --input_as_nhwc true \
    --insert_bn true \
    --precision q8 \
    --image_color bgr \
    --image_std 1/0.00392,1/0.00392,1/0.00392 \
    --add_detect true

