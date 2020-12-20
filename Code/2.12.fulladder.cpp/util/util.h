//
// File name: Util.h
// Created by ronny on 16-7-15.
//

#ifndef NEURALNETWORK_UTIL_H
#define NEURALNETWORK_UTIL_H

#include <string>
#include <opencv2/core/core.hpp>


namespace easyml {

namespace util {

void RandomShuffle(cv::Mat &train_data, cv::Mat &labels);

bool LoadMNIST(const std::string &prefix,
               cv::Mat &train_data,
               cv::Mat &train_labels,
               cv::Mat &test_data,
               cv::Mat &test_labels
);

} // namepsace util
} // namespace easyml

#endif //NEURALNETWORK_UTIL_H
