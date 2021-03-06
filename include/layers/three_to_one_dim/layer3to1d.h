#include <types.h>

#ifndef LAYER_THREE_TO_ONE_DIM_H
#define LAYER_THREE_TO_ONE_DIM_H

using namespace std;

class Layer3to1D
{
public:
	Layer3to1D() {}
	virtual tensor_1d forward(tensor_3d feature_map) = 0;
	virtual tensor_3d backward(tensor_1d gradients) = 0;
};

#endif