#include <tensors/tensor_1d.h>

#ifndef LAYER_ONE_DIM_H
#define LAYER_ONE_DIM_H

using namespace std;

class Layer1D
{
public:
	Layer1D() {}
	virtual tensor_1d forward(tensor_1d inputs) = 0;
	virtual tensor_1d backward(tensor_1d errors_next) = 0;
	virtual void fit(int t, AdamOptimizer& adam) = 0;
};

#endif