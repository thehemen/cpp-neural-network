#include <climits>
#include <types.h>

#ifndef GLOBAL_MAX_POOLING_1D_H
#define GLOBAL_MAX_POOLING_1D_H

using namespace std;

class GlobalMaxPooling1D : public Layer2to1D
{
	int length;
	int width;
	tensor_2d mask;
public:
	GlobalMaxPooling1D(map<string, int> params)
	{
		length = params["count"];
		width = params["width"];
		mask = tensor_2d(length, tensor_1d(width));
	}

	tensor_1d forward(tensor_2d inputs)
	{
		tensor_1d outputs(length);
		make_zero(mask);

		for(int i = 0; i < length; ++i)
		{
			int x_max = 0;
			double value_max = 0.0;

			for(int x = 0; x < width; ++x)
			{
				if(inputs[i][x] > value_max)
				{
					value_max = inputs[i][x];
					x_max = x;
				}
			}

			mask[i][x_max] = 1.0;
			outputs[i] = value_max;
		}

		return outputs;
	}

	tensor_2d backward(tensor_1d gradients)
	{
		tensor_2d gradient_back(length, tensor_1d(width));

		for(int i = 0; i < length; ++i)
		{
			for(int x = 0; x < width; ++x)
			{
				if(mask[i][x])
				{
					gradient_back[i][x] = gradients[i];
				}
			}
		}

		return gradient_back;
	}
};

#endif