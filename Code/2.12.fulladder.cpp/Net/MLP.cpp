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
dGain(1.0)
{
    NumLayers = num_layers;
    pLayers = new Layer<double>[NumLayers];
    for (int i = 0; i < NumLayers; i++){
        pLayers[i].bias = 1.0;
        pLayers[i].NumNeurons = num_Neurons[i];
        pLayers[i].pNeurons = new Neuron<double>[pLayers[i].NumNeurons];
        for (int j = 0; j < num_Neurons[i]; j++){
            // init Neurons params
            pLayers[i].pNeurons[j].output = 1.0;
            if (i>0){
                pLayers[i].pNeurons[j].weight = new double [pLayers[i-1].NumNeurons];
                /* Maybe there can be insert with initialization code
                */
            }else{
                pLayers[i].pNeurons[j].weight = NULL;
            }
        }
    }
    _random_Initialize_weight();
}

MLP::~MLP() {

    for (int i = 0; i < NumLayers; i++){
        if (pLayers[i].pNeurons){
            for (int j = 0; j < pLayers[i].NumNeurons; j++){
                // init Neurons params
                if (pLayers[i].pNeurons){
                    if(pLayers[i].pNeurons[j].weight) delete[] pLayers[i].pNeurons[j].weight;
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
            for (int k = 0; k< pLayers[i-1].NumNeurons; k++)
            pLayers[i].pNeurons[j].weight[k] = u(e);
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
}