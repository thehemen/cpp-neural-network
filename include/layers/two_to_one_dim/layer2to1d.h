#include <tensors/tensor_1d.h>
#include <tensors/tensor_3d.h>

#ifndef LAYER_TWO_TO_ONE_DIM_H
#define LAYER_TWO_TO_ONE_DIM_H

using namespace std;

class Layer2to1D
{
public:
	Layer2to1D() {}
	virtual tensor_1d forward(tensor_3d feature_map) = 0;
	virtual tensor_3d backward(tensor_1d gradients) = 0;
};

#endif