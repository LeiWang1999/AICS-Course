//
// Created by root on 2020/12/23.
//

#include "MLP.h"
#include "cstddef"
#include <random>
#include <math.h>
using namespace std;
MLP::MLP(int num_layers, int *num_Neurons):
NumLayers(0),
pLayers(0),
dGain(1.0),
learning_rate(0.25),
Alpha(0.9)
{
    NumLayers = num_layers;
    pLayers = new Layer<double>[NumLayers];
    output = new double[pLayers[num_layers-1].NumNeurons];
    for (int i = 0; i < NumLayers; i++){
        pLayers[i].bias = 1.0;
        pLayers[i].NumNeurons = num_Neurons[i];
        pLayers[i].pNeurons = new Neuron<double>[pLayers[i].NumNeurons];
        for (int j = 0; j < num_Neurons[i]; j++){
            // init Neurons params
            pLayers[i].pNeurons[j].output = 1.0;
            if (i>0){
                pLayers[i].pNeurons[j].weight = new double [pLayers[i-1].NumNeurons];
                pLayers[i].pNeurons[j].dWeight = new double [pLayers[i-1].NumNeurons];
                /* Maybe there can be insert with initialization code
                */
            }else{
                pLayers[i].pNeurons[j].weight = NULL;
                pLayers[i].pNeurons[j].dWeight = NULL;
            }
        }
    }
    _random_Initialize_weight();
}

MLP::~MLP() {
    if(output) delete [] output;
    for (int i = 0; i < NumLayers; i++){
        if (pLayers[i].pNeurons){
            for (int j = 0; j < pLayers[i].NumNeurons; j++){
                // init Neurons params
                if (pLayers[i].pNeurons){
                    if(pLayers[i].pNeurons[j].weight) delete[] pLayers[i].pNeurons[j].weight;
                    if(pLayers[i].pNeurons[j].dWeight) delete[] pLayers[i].pNeurons[j].dWeight;
                }
            }
            delete[] pLayers[i].pNeurons;
        }
    }
    delete[] pLayers;
}

double MLP::_sigmoid(double x) {
    return 1.0 / (1.0 + exp(-dGain * x));
}

void MLP::_random_Initialize_weight() {
    default_random_engine e;
    uniform_real_distribution<double> u(-1.0, 1.0);
    for (int i = 1; i < NumLayers; i++){
        for (int j = 0; j < pLayers[i].NumNeurons; j++){
            // init Neurons params
            for (int k = 0; k< pLayers[i-1].NumNeurons; k++){
                pLayers[i].pNeurons[j].weight[k] = u(e);
                pLayers[i].pNeurons[j].weight[k] = 0.0;
            }
        }
    }
    return;
}

void MLP::input(double *input_vector) {
    for (int i = 0; i < pLayers[0].NumNeurons; i++){
        pLayers[0].pNeurons[i].output = input_vector[i];
    }
}

void MLP::forward() {
    Layer_Loop:
    for (int i = 1; i < NumLayers; i++){
        Cur_Neurons_Loop:
        for (int j = 0; j < pLayers[i].NumNeurons; j++){
            // init Neurons params
            double sum = 0.0;
            Front_Neurons_Loop:
            for (int k = 0; k < pLayers[i-1].NumNeurons; k++){
                double out = 0.0;
                out = pLayers[i-1].pNeurons[k].output * pLayers[i].pNeurons[j].weight[k];
                sum += out;
            }
            pLayers[i].pNeurons[j].output = _sigmoid(sum);
        }
    }
    for (int i = 0; i < pLayers[NumLayers-1].NumNeurons; ++i) {
        output[i] = pLayers[NumLayers-1].pNeurons[i].output;
    }
}

void MLP::_compute_loss(double *output, double *target) {
    _MSE = 0.0;
    _MAE = 0.0;
    for (int j = 0; j < pLayers[NumLayers - 1].NumNeurons; j++){
        double temp = target[j] - output[j];
        pLayers[NumLayers - 1].pNeurons[j].error = temp;
        _MSE += temp * temp;
        _MAE += fabs(temp);
    }
    _MSE = _MSE / pLayers[NumLayers - 1].NumNeurons;
    _MAE = _MAE / pLayers[NumLayers - 1].NumNeurons;
}

void MLP::backward(double * output, double* target) {
    _compute_loss(output, target);
    _backPropagate();
    _adjust_weights();
}

double MLP::_backPropagate() {
    for (int i = NumLayers-2; i>0 ;i--){
        for (int j = 0; j < pLayers[i].NumNeurons; ++j) {
            double x = pLayers[i].pNeurons[j].output;
            double Error = 0.0;
            for (int k = 0; k < pLayers[i+1].NumNeurons; ++k) {
                Error += pLayers[i+1].pNeurons[k].weight[j] * pLayers[i+1].pNeurons[k].error;
            }
            pLayers[i].pNeurons[j].error = dGain * x * (1.0 - x) * Error;
        }
    }
    return 0;
}

void MLP::_adjust_weights() {
    for(int i = 1; i < NumLayers; i++)
    {
        for(int j = 0; j < pLayers[i].NumNeurons; j++)
        {
            Fronted_Weight_Loop:
            for (int k = 0; k < pLayers[i-1].NumNeurons; k++)
            {
                double x  = pLayers[i-1].pNeurons[k].output;
                double e  = pLayers[i  ].pNeurons[j].error;
                double dw = pLayers[i  ].pNeurons[j].dWeight[k];
                pLayers[i].pNeurons[j].weight[k] += learning_rate * x * e + Alpha * dw;
                pLayers[i].pNeurons[j].dWeight[k]  = learning_rate * x * e;
            }
        }
    }
    cout << "adjust done " << "current loss is " << _MSE << endl;
}

double *MLP::get_output() {
    return output;
}
