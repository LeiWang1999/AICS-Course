#include <iostream>
#include <string>
#include "data/binary_adder.h"
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

void mlp_test(){
    int num_layers = 4;
    int num_Neurons[] = {9, 100, 100, 2};
    double input_vector[] = {0, 1, 1, 0, 1, 1, 0, 0, 1};
    MLP mlp(num_layers, num_Neurons);
    mlp.input(input_vector);
    mlp.forward();
}

int main(){
    mlp_test();
    return 0;
}