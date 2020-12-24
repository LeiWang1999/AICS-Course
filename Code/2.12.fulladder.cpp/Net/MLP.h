//
// Created by root on 2020/12/23.
//

#ifndef INC_2_12_FULLADDER_CPP_MLP_H
#define INC_2_12_FULLADDER_CPP_MLP_H
#include <iostream>

template<class dataType>
class Neuron{
public:
    dataType* weight;
    dataType* dWeight;
    dataType output;
    dataType error;
};

template<class dataTypeBias>
class Layer{
public:
    dataTypeBias bias;
    int NumNeurons;
    Neuron<double> * pNeurons;
};


class MLP {
private:
    int NumLayers;
    double dGain;
    double _MSE;
    double _MAE;
    double learning_rate;
    double Alpha;
    Layer<double>* pLayers;
    double * output;
    void _random_Initialize_weight();
    double _sigmoid(double x);
    void _compute_loss(double output[], double target[]);
    double _backPropagate();
    void _adjust_weights();

public:
    MLP(int layers, int nodes[]);
    ~MLP();
    template <class vectorType>
    void input(vectorType input_vector);
    void input(std::vector<bool> input_vector);
    void input(double * input_vector);
    void forward();
    double get_loss();
    void backward(double * output, double *target);
    double * get_output();
};


#endif //INC_2_12_FULLADDER_CPP_MLP_H
