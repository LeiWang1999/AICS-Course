//
// Created by root on 2020/12/22.
//

#include "data_gen.h"
#include "binary_adder.h"
#include <random>
#include <chrono>
#include <vector>
using namespace std;

template <class RandomAccessIterator, class URNG>
void _shuffle (RandomAccessIterator first, RandomAccessIterator last, URNG&& g)
{
    for (auto i = (last-first) - 1; i > 0; --i) {
        uniform_int_distribution<decltype(i)> d (0,i);
        swap (first[i], first[d (g)]);
    }
}

void x_gen(vector<vector<bool>> &batch_input){
    for (int A_0 = 0; A_0 <= 1; A_0++){
        for (int A_1 = 0; A_1 <= 1; A_1++){
            for (int A_2 = 0; A_2 <= 1; A_2++){
                for (int A_3 = 0; A_3 <= 1; A_3++){
                    for (int B_0 = 0; B_0 <= 1; B_0++){
                        for (int B_1 = 0; B_1 <= 1; B_1++){
                            for (int B_2 = 0; B_2 <= 1; B_2++){
                                for (int B_3 = 0; B_3 <= 1; B_3++) {
                                    for (int C_0 = 0; C_0 <= 1; C_0++) {
                                        vector<bool> x = {(bool)A_0, (bool)A_1, (bool)A_2, (bool)A_3, (bool)B_0,(bool)B_1,(bool)B_2,(bool)B_3,(bool)C_0};
                                        batch_input.push_back(x);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

vector<vector<bool>> y_gen(vector<vector<bool>> &batch_input){
    vector<vector<bool>> batch_output;
    for ( vector<vector<bool>>::const_iterator iter = batch_input.begin(); iter != batch_input.end() ; iter++) {
        vector<bool> element_input = *iter;
        vector<bool> inputA(element_input.begin(), element_input.begin()+4);
        vector<bool> inputB(element_input.begin()+4, element_input.begin()+8);
        bool carry_in = element_input[8];
        vector<bool> result = binary_adder(inputA, inputB, carry_in);
        batch_output.push_back(result);
    }
    return batch_output;
}

void data_gen(vector<vector<bool>> &batch_input, vector<vector<bool>> &batch_output, bool is_shuffle){
    x_gen(batch_input);
    if(is_shuffle){
        unsigned seed = chrono::system_clock::now ().time_since_epoch ().count ();
        _shuffle(batch_input.begin(), batch_input.end(),default_random_engine(seed));
    }
    batch_output = y_gen(batch_input);
}