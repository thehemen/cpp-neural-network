#include <omp.h>  // OpenMP headers
#include <layers/one_dimensional/layer1d.h>
#include <tensors/tensor_1d.h>
#include <tensors/tensor_2d.h>
#include <activation.h>
#include <adam_optimizer.h>

#ifndef DENSE_H
#define DENSE_H

using namespace std;

class Dense : public Layer1D
{
	Activation activation;
	int in_count;
	int out_count;

	tensor_2d weights;
	tensor_1d biases;

	tensor_1d m_t;
	tensor_1d v_t;

	tensor_1d inputs;
	tensor_1d errors;
	tensor_1d args;
	tensor_1d outputs;
public:
	Dense() : Layer1D() {}

	Dense(Activation activation, int in_count, int out_count, tensor_2d weights, tensor_1d biases)
	{
		this->activation = activation;
		this->in_count = in_count;
		this->out_count = out_count;

		this->weights = weights;
		this->biases = biases;

		m_t = tensor_1d(out_count);
		v_t = tensor_1d(out_count);

		inputs = tensor_1d(in_count);
		errors = tensor_1d(out_count);
		args = tensor_1d(out_count);
		outputs = tensor_1d(out_count);
	}

	tensor_1d forward(tensor_1d inputs) override
	{
		this->inputs = inputs;
		outputs.make_zero();

		#pragma omp parallel for
		for(int i = 0; i < out_count; ++i)
		{
			double sum = 0.0;

			for(int j = 0; j < in_count; ++j)
			{
				sum += weights[i][j] * inputs[j];
			}

			sum += biases[i];
			
			args[i] = sum;
			outputs[i] = activation.get(sum);
		}

		return outputs;
	}

	tensor_1d backward(tensor_1d errors_next) override
	{
		errors.make_zero();

		#pragma omp parallel for
		for(int i = 0; i < out_count; ++i)
		{
			errors[i] = activation.der(outputs[i], args[i]) * errors_next[i];
		}

		tensor_1d errors_back(in_count);

		#pragma omp parallel for
		for(int j = 0; j < in_count; ++j)
		{
			double sum = 0.0;

			for(int i = 0; i < out_count; ++i)
			{
				sum += weights[i][j] * errors[i];
			}

			errors_back[j] = sum;
		}

		return errors_back;
	}

	void fit(int t, AdamOptimizer& adam) override
	{
		#pragma omp parallel for
		for(int i = 0; i < out_count; ++i)
		{
			double update = adam.optimize(t, m_t[i], v_t[i], errors[i]);

			for(int j = 0; j < in_count; ++j)
			{
				weights[i][j] += inputs[j] * update;
			}

			biases[i] += update;
		}
	}
};

#endif