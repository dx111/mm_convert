mm_convert \
    --framework pytorch \
    --model models/pytorch_bert/pt_bert_traced.pt \
    --output_model pt_bert_model \
    --archs mtp_372.41 \
    --input_shapes 1,128 1,128 1,128 \
    --pytorch_input_dtypes int32 int32 int32 \
    --precision qint8_mixed_float16 \
    --load_data_func load_squad
