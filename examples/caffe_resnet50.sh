wget -c https://github.com/dx111/models/raw/main/caffe_resnet50/ResNet-50-deploy.prototxt \
    -P models/caffe_resnet50

wget -c https://github.com/dx111/models/raw/main/caffe_resnet50/ResNet-50-model.caffemodel \
    -P models/caffe_resnet50

mm_convert \
    -f caffe \
    --proto models/caffe_resnet50/ResNet-50-deploy.prototxt \
    --model models/caffe_resnet50/ResNet-50-model.caffemodel \
    --output_model caffe_resnet50_model \
    --archs mtp_322 mtp_372.41 \
    --input_shapes 1,3,224,224 \
    --input_as_nhwc true \
    --insert_bn true \
    --precision q8 \
    --image_color bgr \
    --image_mean 103.939002991,116.778999329,123.680000305
