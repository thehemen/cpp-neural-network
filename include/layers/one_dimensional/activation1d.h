#include <omp.h>  // OpenMP headers
#include <layers/one_dimensional/layer1d.h>
#include <ops.h>
#include <types.h>
#include <activation.h>

#ifndef ACTIVATION_1D_H
#define ACTIVATION_1D_H

using namespace std;

class Activation1D : public Layer1D
{
	Activation activation;

	int out_count;
	tensor_1d inputs;
	tensor_1d outputs;

	tensor_1d gradients;
	tensor_1d gradients_back;
public:
	Activation1D(Activation activation, map<string, int> params)
	{
		this->activation = activation;
		out_count = params["out_count"];

		inputs = tensor_1d(out_count);
		outputs = tensor_1d(out_count);

		gradients = tensor_1d(out_count);
		gradients_back = tensor_1d(out_count);
	}

	tensor_1d forward(tensor_1d inputs) override
	{
		this->inputs = inputs;
		make_zero(outputs);

		#pragma omp parallel for
		for(int i = 0; i < out_count; ++i)
		{
			outputs[i] = activation.get(inputs[i]);
		}

		return outputs;
	};

	tensor_1d backward(tensor_1d gradients) override
	{
		this->gradients = gradients;
		make_zero(gradients_back);

		#pragma omp parallel for
		for(int i = 0; i < out_count; ++i)
		{
			gradients_back[i] = activation.der(outputs[i], inputs[i]) * gradients[i];
		}

		return gradients_back;
	};

	void fit(int t, AdamOptimizer& adam) override {}
};

#endif