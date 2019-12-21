#include <tensors/tensor_2d.h>

#ifndef LAYER_TWO_DIM_H
#define LAYER_TWO_DIM_H

using namespace std;

class Layer2D
{
public:
	Layer2D() {}
	virtual tensor_3d forward(tensor_3d feature_map) = 0;
	virtual tensor_3d backward(tensor_3d gradients) = 0;
	virtual void fit(int t, AdamOptimizer& adam) = 0;
};

#endif