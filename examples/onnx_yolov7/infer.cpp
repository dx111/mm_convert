#include <cnrt.h>
#include <mm_runtime.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <utility>
#include <cassert>

size_t img_size[] = {640, 640};
std::string model_file = "yolov7.mm";
std::vector<std::string> image_file_list{
    "./data/zidane.jpg",
    "./data/bus.jpg"
};

cv::Mat process_img(cv::Mat src_img){
    int src_h = src_img.rows;
    int src_w = src_img.cols;
    int dst_h = img_size[0];
    int dst_w = img_size[1];
    float ratio = std::min(float(dst_h)/float(src_h), float(dst_w)/float(src_w));
    int unpad_h = int(std::floor(src_h * ratio));
    int unpad_w = int(std::floor(src_w * ratio));
    if(ratio !=1){
        int interpolation;
        if(ratio < 1){
            interpolation = cv::INTER_AREA;
        }else{
            interpolation = cv::INTER_LINEAR;
        }
        cv::resize(src_img, src_img, cv::Size(unpad_w, unpad_h), interpolation);
    }

    int pad_t = std::floor((dst_h - unpad_h)/2);
    int pad_b = dst_h - unpad_h - pad_t;
    int pad_l = std::floor((dst_w - unpad_w)/2);
    int pad_r = dst_w - unpad_w - pad_l;

    cv::copyMakeBorder(src_img, src_img, pad_t, pad_b, pad_l, pad_r, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return src_img;
}


int main()
{
    auto model = magicmind::CreateIModel();
    model->DeserializeFromFile(model_file.c_str());

    size_t batch_size = model->GetInputDimension(0).GetDimValue(0);

    std::vector<cv::Mat> img_list;
    std::vector<cv::Mat> show_img_list;
    for(auto file: image_file_list){
        cv::Mat img = cv::imread(file);
        show_img_list.push_back(img.clone());
        img = process_img(img);
        img_list.push_back(img);
    }

    // 1 create queue
    cnrtQueue_t queue;
    assert(cnrtQueueCreate(&queue) == cnrtSuccess);

    // 2.crete engine
    auto engine = model->CreateIEngine();
    assert(engine != nullptr);
  
    // 3.create context
    auto context = engine->CreateIContext();
    assert(context != nullptr);

    // 5.crete input tensor and output tensor and memory alloc
    std::vector<magicmind::IRTTensor *> input_tensors, output_tensors;
    assert(context->CreateInputTensors(&input_tensors).ok());
    assert(context->CreateOutputTensors(&output_tensors).ok());
    assert(context->InferOutputShape(input_tensors, output_tensors).ok());

    // input tensor memory alloc
    for (auto tensor : input_tensors)
    {
        void *mlu_addr_ptr;
        assert(cnrtMalloc(&mlu_addr_ptr, tensor->GetSize()) == cnrtSuccess);
        assert(tensor->SetData(mlu_addr_ptr).ok());
    }

    // output tensor memory alloc
    for (auto tensor : output_tensors)
    {
        void *mlu_addr_ptr;
        assert(cnrtMalloc(&mlu_addr_ptr, tensor->GetSize()) == cnrtSuccess);
        assert(tensor->SetData(mlu_addr_ptr).ok());
    }

    // 6. copy in
    size_t size_per_batch = input_tensors[0]->GetSize() / batch_size;
    for(int i = 0 ; i < img_list.size() ; ++i){
        void* mlu_ptr = input_tensors[0]->GetMutableData() + size_per_batch * i;
        void* cpu_ptr = img_list[i].data;
        cnrtMemcpy(mlu_ptr, cpu_ptr, size_per_batch, CNRT_MEM_TRANS_DIR_HOST2DEV);
    }

    // 7. compute
    assert(context->Enqueue(input_tensors, output_tensors, queue).ok());
    assert(cnrtQueueSync(queue) == cnrtSuccess);

    // 8. copy out
    std::vector<void *> result;
    for (uint32_t i = 0; i < output_tensors.size(); ++i){
        void *memory = malloc(output_tensors[i]->GetSize());
        cnrtMemcpy(memory, output_tensors[i]->GetMutableData(), output_tensors[i]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST);
        result.push_back(memory);
    }
    
    // 9. postprocess
    std::vector<int> box_num((int*)result[1], ((int*)result[1] + batch_size));
    for(int b=0 ; b < batch_size ; ++b){
        float* box_ptr = (float*)result[0] + b * 7 * 2048;
        cv::Mat show_img = show_img_list[b];
        int src_h = show_img.rows;
        int src_w = show_img.cols;
        int dst_h = 640;
        int dst_w = 640;
        float ratio = std::min(float(dst_h)/float(src_w), float(dst_w)/float(src_h));
        for( int i = 0 ; i < box_num[b] ; ++i){
            float score = *(box_ptr + 2 + (i * 7));
            int detect_class = *(box_ptr + 1 + (i * 7));
            float xmin = *(box_ptr + 3 + (i * 7));
            float ymin = *(box_ptr + 4 + (i * 7));
            float xmax = *(box_ptr + 5 + (i * 7));
            float ymax = *(box_ptr + 6 + (i * 7));

             int unpad_h = int(std::floor(src_h * ratio));
            int unpad_w = int(std::floor(src_w * ratio));

            int pad_t = std::floor((dst_h - unpad_h)/2);
            int pad_l = std::floor((dst_w - unpad_w)/2);

            xmin = (xmin - pad_l) / ratio;
            ymin = (ymin - pad_t) / ratio;
            xmax = (xmax - pad_l) / ratio;
            ymax = (ymax - pad_t) / ratio;
            cv::rectangle(show_img, cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)), cv::Scalar(0, 255, 0));
        }
        cv::imwrite("cpp_res_" + std::to_string(b) + ".jpg" , show_img);
    }

    // free cpu ptr
    for(auto ptr: result){
        free(ptr);
    }

    // 10. destroy mlu memory
    for (auto tensor : input_tensors)
    {
        cnrtFree(tensor->GetMutableData());
        tensor->Destroy();
    }
    for (auto tensor : output_tensors)
    {
        cnrtFree(tensor->GetMutableData());
        tensor->Destroy();
    }
    context->Destroy();
    engine->Destroy();
    return 0;
}