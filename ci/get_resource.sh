#!/bin/bash
set -e

ftp_user=${FTP_USER}
ftp_password=${FTP_PASSWORD}

declare -A metadata
metadata["arcface_pytorch16"]="models/pytorch_arcface/arcface_r100.pt"
metadata["bert_pytorch"]="models/pytorch_bert/bert_mask.pt"
metadata["c3d_pytorch16"]="models/pytorch_c3d/py3.7.9_torch1.6.0_c3d.pt"
metadata["deeplabv3+_pytorch"]="dataset/voc2007_1000.tar.gz,models/pytorch_deeplabv3+/py3.7.9_torch1.6.0_deeplabv3+.pt"
metadata["efficientnet-b0_pytorch16"]="dataset/imagenet1000.tar.gz,models/pytorch_efficientnet/py3.7.9_torch1.6.0_efficientnet-b0.pt"
metadata["refinedet_caffe"]="dataset/voc2007_1000.tar.gz,models/caffe_refinedet/refinedet_vgg16.prototxt,models/caffe_refinedet/refinedet_vgg16.caffemodel"
metadata["retinaface_pytorch"]="/dataset/fddb_images.zip,models/pytorch_retinaface/retinaface_mobilenet.pt"
metadata["segnet_caffe"]="dataset/voc2007_1000.tar.gz,models/caffe_segnet/segnet_pascal.prototxt,models/caffe_segnet/segnet_pascal.caffemodel"
metadata["ssd_caffe"]="dataset/voc2007_1000.tar.gz,models/caffe_ssd/ssd.prototxt,models/caffe_ssd/ssd.caffemodel"
metadata["u2net_pytorch16"]="dataset/MSRA-B.tar.gz,models/pytorch_u2net/u2net.pt"
metadata["yolov3_caffe"]="dataset/coco1000.tar.gz,models/caffe_yolov3/yolov3_416.prototxt,models/caffe_yolov3/yolov3_416.caffemodel"
metadata["yolov3-tiny_tf"]="dataset/coco1000.tar.gz,models/tf_yolov3-tiny/frozen_darknet_yolov3_model.pbtxt"
metadata["yolov4_caffe"]="dataset/voc2007_1000.tar.gz,models/caffe_yolov4/yolov4.prototxt,models/caffe_yolov4/yolov4.caffemodel"
metadata["yolov5_v5.0_pytorch16"]="dataset/coco1000.tar.gz,models/pytorch_yolov5m/yolov5m_v5.pt"
metadata["deeplabv3_tf"]="dataset/VOCtrainval_11-May-2012.tar,models/tf_deeplabv3/frozen_inference_graph.pb"
metadata["resnet101_tf"]="dataset/imagenet1000.tar.gz,models/tf_resnet101_v1/frozen_resnet_v1_101.pb"
metadata["yolov5_v5.0_pytorch16"]="dataset/coco1000.tar.gz,models/pytorch_yolov5x_v2/yolov5x.torchscript.pt"
metadata["resnet50_pytorch"]="dataset/imagenet1000.tar.gz,models/pytorch_resnet50/py3.7.9_torch1.6.0_resnet50.pt"
metadata["yolop_onnx"]="models/onnx_yolop/yolop-640-640.onnx"
metadata["resnet50_onnx"]="dataset/imagenet1000.tar.gz,models/onnx_resnet50/resnet50-v1-7.onnx"
metadata["yolov5x_v6.1_onnx"]="dataset/coco1000.tar.gz,models/onnx_yolov5_v6.1/yolov5x.onnx"
metadata["densent121_onnx"]="dataset/imagenet1000.tar.gz,models/onnx_densent121/densenet-12.onnx"

if [ $# == 0 ] ; then
    download_key=${!metadata[*]}
else
    download_key=$@
fi

for var in ${download_key[@]}
do
    _exist=0
    for k in ${!metadata[*]}
    do
        if [[ "$var" == "$k" ]]; then
            _exist=1
        fi
    done

    if [[ $_exist -eq 1 ]];then
        resource=${metadata["$var"]}
        array=(${resource//,/ }) 
        for file in ${array[@]}
        do
        wget -N -r -c -nH -np \
            -P ../../ \
            --cut-dirs=2 \
            --ftp-user=${ftp_user} \
            --ftp-password=${FTP_PASSWORD} \
            ftp://download.cambricon.com:8821/product/datasets/cnbox_resource/${file}
        done
    else
        echo "args ${var} not found, support args: ${!metadata[*]}"
    fi
done


# decompress dataset
pushd ../../cnbox_resource/dataset

for file in $(ls .)
do
    if [[ "${file}" = *.tar.gz ]]; then
        tar -xf $file --skip-old-files
    elif [[ "${file}" = *.tar ]]; then
        tar -xf $file --skip-old-files
    elif [[ "${file}" = *zip ]]; then
        unzip  -n $file
    fi
done

popd
