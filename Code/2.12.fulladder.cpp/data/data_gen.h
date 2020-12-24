//
// Created by root on 2020/12/22.
//

#ifndef INC_2_12_FULLADDER_CPP_DATA_GEN_H
#define INC_2_12_FULLADDER_CPP_DATA_GEN_H
#include <vector>
using namespace std;
void data_gen(std::vector<std::vector<bool>> &batch_input, std::vector<std::vector<bool>> &batch_output, bool is_shuffle);
void data_split(vector<vector<bool>> &batch_input, vector<vector<bool>> &batch_output,
                vector<vector<bool>> &train_input, vector<vector<bool>> &train_output,
                vector<vector<bool>> &test_input, vector<vector<bool>> &test_output, double rate);
#endif //INC_2_12_FULLADDER_CPP_DATA_GEN_H
