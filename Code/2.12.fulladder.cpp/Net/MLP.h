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
    dataType output;
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
    Layer<double>* pLayers;
    void _random_Initialize_weight();
    double _sigmoid(double x);
public:
    MLP(int layers, int nodes[]);
    ~MLP();
    void input(double * input_vector);
    void forward();
};


#endif //INC_2_12_FULLADDER_CPP_MLP_H
