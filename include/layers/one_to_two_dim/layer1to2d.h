#include <types.h>

#ifndef LAYER_ONE_TO_TWO_DIM_H
#define LAYER_ONE_TO_TWO_DIM_H

using namespace std;

class Layer1to2D
{
public:
	Layer1to2D() {}
	virtual tensor_2d forward(tensor_1d inputs) = 0;
	virtual tensor_1d backward(tensor_2d gradients) = 0;
	virtual void fit(int t, AdamOptimizer& adam) = 0;
};

#endif