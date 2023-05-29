#include <cnrt.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <opencv2/opencv.hpp>

#include <mm_runtime.h>

std::string model_path = "onnx_resnet50_model";
std::string img_file = "bus.jpg";
int img_size[2] = {224, 224}; // w, h

template <typename T>
void print_data(void *ptr, int num = 10)
{
    T *cpu_ptr = (T *)malloc(num * sizeof(T));
    cnrtMemcpy(cpu_ptr, ptr, num * sizeof(T), cnrtMemcpyDevToHost);
    for (int i = 0; i < num; ++i){
        std::cout << *(cpu_ptr + i) << ", ";
    }
    std::cout << std::endl;
}

int main(){
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    auto model = magicmind::CreateIModel();
    model->DeserializeFromFile(model_path.c_str());
    auto engine = model->CreateIEngine();
    auto context = engine->CreateIContext();

    std::vector<magicmind::IRTTensor *> input_tensors, output_tensors;
    context->CreateInputTensors(&input_tensors);
    context->CreateOutputTensors(&output_tensors);

    context->InferOutputShape(input_tensors, output_tensors);

    for (auto tensor : input_tensors){
        void *mlu_ptr = nullptr;
        cnrtMalloc(&mlu_ptr, tensor->GetSize());
        tensor->SetData(mlu_ptr);
    }

    for (auto tensor : output_tensors){
        void *mlu_ptr = nullptr;
        cnrtMalloc(&mlu_ptr, tensor->GetSize());
        tensor->SetData(mlu_ptr);
    }

    // load img
    cv::Mat img = cv::imread(img_file);
    cv::resize(img, img, cv::Size(img_size[0], img_size[1]));
    img.convertTo(img, CV_32F);

    // copy in 
    cnrtMemcpy(input_tensors[0]->GetMutableData(), img.data, input_tensors[0]->GetSize(), cnrtMemcpyHostToDev);
    
    print_data<float>(input_tensors[0]->GetMutableData());

    // compute
    context->Enqueue(input_tensors, output_tensors, queue);
    CNRT_CHECK(cnrtQueueSync(queue));

    // compy out
    void* cpu_ptr = malloc(output_tensors[0]->GetSize());
    cnrtMemcpy(cpu_ptr, output_tensors[0]->GetMutableData(), output_tensors[0]->GetSize(), cnrtMemcpyDevToHost);

    print_data<float>(output_tensors[0]->GetMutableData());

    return 0;

}