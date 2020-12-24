//
// Created by root on 2020/12/22.
//

#include "data_gen.h"
#include "binary_adder.h"

using namespace std;
void x_gen(vector<vector<int>> &batch_input){
    for (int A_0 = 0; A_0 <= 1; A_0++){
        for (int A_1 = 0; A_1 <= 1; A_1++){
            for (int A_2 = 0; A_2 <= 1; A_2++){
                for (int A_3 = 0; A_3 <= 1; A_3++){
                    for (int B_0 = 0; B_0 <= 1; B_0++){
                        for (int B_1 = 0; B_1 <= 1; B_1++){
                            for (int B_2 = 0; B_2 <= 1; B_2++){
                                for (int B_3 = 0; B_3 <= 1; B_3++) {
                                    for (int C_0 = 0; C_0 <= 1; C_0++) {
                                        vector<int> x = {A_0, A_1,A_2,A_3,B_0,B_1,B_2,B_3,C_0};
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

void y_gen(vector<vector<int>> &batch_input, vector<vector<int>> &batch_output){
    
}

void data_gen(vector<vector<int>> &batch_input, vector<vector<int>> &batch_output){
    x_gen(batch_input);
    y_gen(batch_input, batch_output);
}