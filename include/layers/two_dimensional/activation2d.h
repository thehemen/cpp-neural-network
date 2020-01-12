#include <map>
#include <omp.h>  // OpenMP headers
#include <ops.h>
#include <types.h>
#include <layers/two_dimensional/layer2d.h>

#ifndef ACTIVATION_2D_H
#define ACTIVATION_2D_H

using namespace std;

class Activation2D : public Layer2D
{
	Activation activation;

	int length;
	int width;

	tensor_2d inputs;
	tensor_2d outputs;
public:
	Activation2D(Activation activation, map<string, int> params)
	{
		this->activation = activation;

		length = params["count"];
		width = params["width"];

		inputs = tensor_2d(length, tensor_1d(width));
		outputs = tensor_2d(length, tensor_1d(width));
	}

	tensor_2d forward(tensor_2d inputs) override
	{
		this->inputs = inputs;
		make_zero(outputs);

		#pragma omp parallel for
		for(int i = 0; i < length; ++i)
		{
			for(int x = 0; x < width; ++x)
			{
				outputs[i][x] = activation.get(inputs[i][x]);
			}
		}

		return outputs;
	};

	tensor_2d backward(tensor_2d gradients) override
	{
		tensor_2d gradients_back = tensor_2d(length, tensor_1d(width));

		#pragma omp parallel for
		for(int i = 0; i < length; ++i)
		{
			for(int x = 0; x < width; ++x)
			{
				gradients_back[i][x] = activation.der(outputs[i][x], inputs[i][x]) * gradients[i][x];
			}
		}

		return gradients_back;
	};

	void fit(int t, AdamOptimizer& adam) override {};
};

#endif