//
// File name: fully_connected_layer.h
// Created by ronny on 16-12-6.
//

#ifndef EASYML_NERUALNETWORK_INPUT_LAYER_H
#define EASYML_NERUALNETWORK_INPUT_LAYER_H


#include "common.h"
#include "layer.h"

namespace easyml {
namespace nn {

class InputLayerParameter: public LayerParameter
{
public:
    InputLayerParameter(const std::string &layer_name)
        :LayerParameter(layer_name, LAYER_INPUT)
    {

    }
};



class InputLayer : public Layer
{
public:
    explicit InputLayer(const InputLayerParameter &param);

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

} // namespace easyml
} // namespace nn

#endif // EASYML_NERUALNETWORK_INPUT_LAYER_H


