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
    vector<bool> v_a = {1,0,0,1};
    vector<bool> v_b = {1};
    vector<bool> v_result;
    bool v_carry = 0;
    v_result = binary_adder(v_a, v_b, carry);
    return;
}

void mlp_test(){
    int num_layers = 4;
    int num_Neurons[] = {9, 100, 100, 5};
    double input_vector[] = {0, 1, 1, 0, 1, 1, 0, 0, 1};
    double target[] = {0,0,1,0,1};
    double* output = NULL;
    MLP mlp(num_layers, num_Neurons);
    // forward test
    for (int i = 0; i < 10; ++i) {
        cout << "Current Loop : " << i << endl;
        mlp.input<double *>(input_vector);
        mlp.forward();
        output = mlp.get_output();
        mlp.backward(output, target);
    }
}

int main(){
    // generate test data
    vector<vector<bool>> batch_input;
    vector<vector<bool>> batch_output;
    vector<vector<bool>> train_input;
    vector<vector<bool>> train_output;
    vector<vector<bool>> test_input;
    vector<vector<bool>> test_output;
    data_gen(batch_input, batch_output, true);
    data_split(batch_input, batch_output,
            train_input,train_output,
            test_input, test_output,
            0.7
            );
    // init MLP
    int num_layers = 4;
    int num_Neurons[] = {9, 20, 20, 5};
    int epochs = 200;
    MLP mlp(num_layers, num_Neurons);
    for(int epoch = 1; epoch <= epochs; epoch++){
        cout << "Current Epoch : " << epoch << " Batch Size : " << train_input.size() ;
        for (int i = 0; i < train_input.size(); ++i) {
            double* output = NULL;
            double target[num_Neurons[num_layers - 1]] ;
            mlp.input(train_input[i]);
            mlp.forward();
            output = mlp.get_output();
            for (int j = 0; j < num_Neurons[num_layers-1]; ++j) {
                target[j] = train_output[i][j];
            }
            mlp.backward(output, target);
        }
        int acc = 0;
        cout << " training loss : " << mlp.get_loss() << " outputBatchSize : " << test_input.size();
        for (int i = 0; i < test_input.size(); ++i) {
            double* output = NULL;
            mlp.input(test_input[i]);
            mlp.forward();
            output = mlp.get_output();
            // loop neurons
            int j;
            for (j = 0; j < num_Neurons[num_layers-1]; ++j) {
                if ((output[j]<0.5 && test_output[i][j] == 0) || (output[j] >= 0.5 && test_output[i][j] == 1))
                    continue;
                else
                    break;
            }
            if(j == num_Neurons[num_layers-1]){
                acc++;
            }
        }
        cout << " acc : " << 1.0 * acc / test_input.size() << endl;
    }
//    data_gen_test();
//    mlp_test();

    return 0;
}