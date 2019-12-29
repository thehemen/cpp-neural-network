#include <omp.h>  // OpenMP headers
#include <layers/one_dimensional/layer1d.h>
#include <ops.h>
#include <types.h>
#include <adam_optimizer.h>

#ifndef DENSE_H
#define DENSE_H

using namespace std;

class Dense : public Layer1D
{
	int input_count;
	int out_count;

	tensor_2d weights;
	tensor_1d biases;

	tensor_1d m_t;
	tensor_1d v_t;

	tensor_1d inputs;
	tensor_1d outputs;

	tensor_1d gradients;
	tensor_1d gradients_back;
public:
	Dense() : Layer1D() {}

	Dense(tensor_2d weights, tensor_1d biases, map<string, int> params)
	{
		this->weights = weights;
		this->biases = biases;

		input_count = params["input_count"];
		out_count = params["length"];

		m_t = tensor_1d(out_count);
		v_t = tensor_1d(out_count);

		inputs = tensor_1d(input_count);
		outputs = tensor_1d(out_count);

		gradients = tensor_1d(out_count);
		gradients_back = tensor_1d(input_count);
	}

	tensor_1d forward(tensor_1d inputs) override
	{
		this->inputs = inputs;
		make_zero(outputs);

		#pragma omp parallel for
		for(int i = 0; i < out_count; ++i)
		{
			double sum = 0.0;

			for(int j = 0; j < input_count; ++j)
			{
				sum += weights[i][j] * inputs[j];
			}

			sum += biases[i];
			outputs[i] = sum;
		}

		return outputs;
	}

	tensor_1d backward(tensor_1d gradients) override
	{
		this->gradients = gradients;
		make_zero(gradients_back);

		#pragma omp parallel for
		for(int j = 0; j < input_count; ++j)
		{
			double sum = 0.0;

			for(int i = 0; i < out_count; ++i)
			{
				sum += weights[i][j] * gradients[i];
			}

			gradients_back[j] = sum;
		}

		return gradients_back;
	}

	void fit(int t, AdamOptimizer& adam) override
	{
		#pragma omp parallel for
		for(int i = 0; i < out_count; ++i)
		{
			double update = adam.optimize(t, m_t[i], v_t[i], gradients[i]);

			for(int j = 0; j < input_count; ++j)
			{
				weights[i][j] += inputs[j] * update;
			}

			biases[i] += update;
		}
	}
};

#endif