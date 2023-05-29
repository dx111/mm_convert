wget -nc -c https://github.com/dx111/models/raw/main/tf_deeplabv3/frozen_inference_graph.pb

mm_convert \
    --framework tensorflow \
    --model frozen_inference_graph.pb \
    --tf_model_type tf-graphdef-file \
    --tf_graphdef_inputs ImageTensor:0 \
    --tf_graphdef_outputs SemanticPredictions:0 \
    --output_model tensorflow_deeplabv3p_model \
    --archs mtp_372 \
    --input_shapes 1,513,513,3 \
    --precision qint8_mixed_float16 \
    --image_color rgb \
