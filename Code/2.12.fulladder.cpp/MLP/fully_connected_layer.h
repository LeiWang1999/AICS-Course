//
// File name: fully_connected_layer.h
// Created by ronny on 16-12-2.
//

#ifndef EASYML_NERUALNETWORK_FULLY_CONNECTED_LAYER_H
#define EASYML_NERUALNETWORK_FULLY_CONNECTED_LAYER_H


#include <easyml/common.h>
#include <easyml/neural_network/layer.h>


namespace easyml {
namespace nn {

class FullyConnectedLayerParameter : public LayerParameter
{
public:
    FullyConnectedLayerParameter(
            const std::string &layer_name,
            Dim in_dim,
            Dim out_dim,
            std::shared_ptr<util::ActivationFunction> activation_fun)
            :LayerParameter(layer_name, LAYER_FC, activation_fun)
    {
        input_dim = in_dim;
        output_dim = out_dim;
    }
    Dim input_dim;
    Dim output_dim;
};


class FullyConnectedLayer : public Layer
{
public:

    explicit FullyConnectedLayer(const FullyConnectedLayerParameter &param);

    /// @brief forward computation
    void FeedForward(
            const std::vector<cv::Mat> &input,
            std::vector<cv::Mat> &output) override;


    /// @brief the loss propagate back throung the layer
    void BackPropagation(
            const std::vector<cv::Mat> &delta_in,
            std::vector<cv::Mat> &delta_out,
            float eta,
            float lambda) override;

    void SetLabels(const std::vector<cv::Mat> &labels) { };
};


} // namespace nn
} // namespace easyml


#endif // EASYML_NERUALNETWORK_FULLY_CONNECTED_LAYER_H

