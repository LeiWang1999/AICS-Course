#include "fully_connected_layer.h"
#include <glog/logging.h>

namespace easyml {
namespace nn {

FullyConnectedLayer::FullyConnectedLayer(const FullyConnectedLayerParameter &param)
{
    name_ = param.name;
    type_ = param.type;

    biases_ = cv::Mat(param.output_dim.height, 1, CV_32FC1);
    cv::randn(biases_, cv::Scalar::all(0.0f), cv::Scalar::all(1.0f));

    weights_ = cv::Mat(param.output_dim.height, param.input_dim.height, CV_32FC1);
    float init_weight = 1.0f / sqrt(static_cast<float>(param.input_dim.height));
    cv::randn(weights_, cv::Scalar::all(0.0f), cv::Scalar::all(init_weight));

    activation_ = param.activation;

}


void FullyConnectedLayer::FeedForward(
        const std::vector<cv::Mat> &input,
        std::vector<cv::Mat> &output)
{
    input_.assign(input.size(), cv::Mat());

    int batch_size = input_.size();
    output.assign(batch_size, cv::Mat());
    weighted_output_.assign(batch_size, cv::Mat());

    for (int i = 0; i < batch_size; i++) {
        input_[i] = input[i].clone();
        weighted_output_[i] = weights_ * input_[i] + biases_;
        output[i] = (*activation_)(weighted_output_[i]);
    }
}

void FullyConnectedLayer::BackPropagation(
        const std::vector<cv::Mat> &delta_in,
        std::vector<cv::Mat> &delta_out,
        float eta,
        float lambda)
{
    cv::Mat nabla_w_sum(weights_.size(), CV_32FC1, cv::Scalar(0.0f));
    cv::Mat nabla_b_sum(biases_.size(), CV_32FC1, cv::Scalar(0.0f));

    int batch_size = input_.size();
    delta_out.assign(batch_size, cv::Mat());

    for (int i = 0; i < batch_size; i++) {
        delta_out[i] = delta_in[i].mul(activation_->primer(weighted_output_[i]));
        cv::Mat nabla_b = delta_out[i].clone();
        cv::Mat nabla_w = nabla_b * input_[i].t();
        nabla_w_sum += nabla_w;
        nabla_b_sum += nabla_b;
        delta_out[i] = weights_.t() * delta_out[i];
    }
    weights_ *= (1 - eta * lambda); // L2 regularization
    weights_ -= (eta / batch_size) * nabla_w_sum;
    biases_ -= (eta / batch_size) * nabla_b_sum;
}

} // namespace nn
} // namespace easyml
