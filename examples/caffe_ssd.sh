wget -c https://github.com/dx111/models/raw/main/caffe_ssd/ssd.caffemodel \
   -P models/caffe_ssd

wget -c https://github.com/dx111/models/raw/main/caffe_ssd/ssd.prototxt \
   -P models/caffe_ssd

mm_convert \
    --framework caffe \
    --proto models/caffe_ssd/ssd.prototxt \
    --model models/caffe_ssd/ssd.caffemodel \
    --output_model caffe_ssd_model \
    --archs mtp_372 \
    --input_shapes 1,3,300,300 \
    --input_as_nhwc true \
    --insert_bn true \
    --precision q8 \
    --image_color bgr \
    --image_mean 127.5,127.5,127.5 \
    --image_std 1/0.007843,1/0.007843,1/0.007843
