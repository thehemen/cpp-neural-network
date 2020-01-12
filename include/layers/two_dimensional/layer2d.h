#include <types.h>

#ifndef LAYER_TWO_DIM_H
#define LAYER_TWO_DIM_H

using namespace std;

class Layer2D
{
public:
	Layer2D() {}
	virtual tensor_2d forward(tensor_2d inputs) = 0;
	virtual tensor_2d backward(tensor_2d gradients) = 0;
	virtual void fit(int t, AdamOptimizer& adam) = 0;
};

#endif