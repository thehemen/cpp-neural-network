#include <map>
#include <omp.h>
#include <types.h>
#include <layers/two_dimensional/layer2d.h>

#ifndef SPATIAL_DROPOUT_1D_H
#define SPATIAL_DROPOUT_1D_H

using namespace std;

class SpatialDropout1D : public Layer2D
{
	int length;
	int width;
	double share;
public:
	SpatialDropout1D(map<string, float> params)
	{
		length = params["count"];
		width = params["width"];
		share = params["share"];
	}

	tensor_2d forward(tensor_2d inputs) override
	{
		return inputs;
	}

	tensor_2d backward(tensor_2d gradients) override
	{
		tensor_2d gradient_back(length, tensor_1d(width));

		#pragma omp parallel for
		for(int i = 0; i < length; ++i)
		{
			// When random < share, a gradient feature map is not backpropagated.
			if(get_random_value() < share)
			{
				continue;
			}

			for(int x = 0; x < width; ++x)
			{
				gradient_back[i][x] = gradients[i][x];
			}
		}

		return gradient_back;
	}

	void fit(int t, AdamOptimizer& adam) override {}
};

#endif