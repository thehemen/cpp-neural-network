#include <vector>

#ifndef TYPES_H
#define TYPES_H

typedef std::vector<double> tensor_1d;
typedef std::vector<tensor_1d> tensor_2d;
typedef std::vector<tensor_2d> tensor_3d;
typedef std::vector<tensor_3d> tensor_4d;

struct Sample1D
{
    tensor_1d inputs;
    tensor_1d outputs;

    Sample1D(tensor_1d inputs, tensor_1d outputs)
    {
        this->inputs = inputs;
        this->outputs = outputs;
    }
};

struct Sample3to1D
{
	tensor_3d inputs;
	tensor_1d outputs;

	Sample3to1D(tensor_3d inputs, tensor_1d outputs)
	{
		this->inputs = inputs;
		this->outputs = outputs;
	}
};

#endif