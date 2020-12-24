#include <iostream>
#include <string>
#include <vector>
#include "data/binary_adder.h"
#include "data/data_gen.h"
#include "Net/MLP.h"

using namespace std;

void data_gen_test(){
    string a = "1001";
    string b = "1";
    bool carry = 0;
    string result = "0";
    result = binary_adder(a , b, carry);
    cout << result << endl;
}

//void mlp_test(){
//    int num_layers = 4;
//    int num_Neurons[] = {9, 100, 100, 5};
//    double input_vector[] = {0, 1, 1, 0, 1, 1, 0, 0, 1};
//    double target[] = {0,0,1,0,1};
//    double* output = NULL;
//    MLP mlp(num_layers, num_Neurons);
//    // forward test
//    for (int i = 0; i < 10; ++i) {
//        cout << "Current Loop : " << i << endl;
//        mlp.input(input_vector);
//        mlp.forward();
//        output = mlp.get_output();
//        mlp.backward(output, target);
//    }
//}

int main(){
    vector<vector<int>> batch_input;
    vector<vector<int>> batch_output;
    data_gen(batch_input, batch_output);


//    data_gen_test();
//    mlp_test();

    return 0;
}