//
// File name: fully_connected_layer.h
// Created by ronny on 16-12-6.
//

#include <easyml/neural_network/input_layer.h>

namespace easyml {
namespace nn {

InputLayer::InputLayer(const InputLayerParameter &param)
{
    name_ = param.name;
    type_ = param.type;
}

void InputLayer::FeedForward(const std::vector<cv::Mat> &input, std::vector<cv::Mat> &output)
{
    output.assign(input.size(), cv::Mat());
    for (size_t i = 0; i < input.size(); i++)
    {
        output[i] = input[i].clone();
    }
}


void InputLayer::BackPropagation(
            const std::vector<cv::Mat> &delta_in,
            std::vector<cv::Mat> &delta_out,
            float eta,
            float lambda)
{
}

} // namespace easyml
} // namespace nn



