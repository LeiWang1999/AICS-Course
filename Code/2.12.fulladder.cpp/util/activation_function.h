//
// File name: activation_function.h
// Created by ronny on 16-12-2.
//

#ifndef EASYML_NEURALNETWORK_ACTIVATIONFUNCTION_H
#define EASYML_NEURALNETWORK_ACTIVATIONFUNCTION_H

namespace easyml {

namespace util {


class ActivationFunction
{
public:
    ActivationFunction()  = default;

    virtual inline cv::Mat operator()(const cv::Mat& input) 
    { 
        return input.clone();
    }

    virtual inline cv::Mat primer(const cv::Mat &input)
    {
        return cv::Mat::ones(input.size(), input.type());
    }

    virtual ~ActivationFunction() = default;
};

class SigmoidFunction : public ActivationFunction
{
public:
    inline cv::Mat operator()(const cv::Mat& input) override
    {
        cv::Mat expon;
        cv::exp(-input, expon);
        return 1.0f / (1.0f + expon);
    }
    inline cv::Mat primer(const cv::Mat &input) override
    {
        cv::Mat a = (*this)(input);
        return a.mul(1.0f - a);
    }
    ~SigmoidFunction() { }
};

} // namepsace util
} // namespace easyml


#endif // EASYML_NEURALNETWORK_ACTIVATIONFUNCTION_H
