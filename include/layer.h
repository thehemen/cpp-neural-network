#include <omp.h>  // OpenMP headers
#include <vec.h>
#include <activation.h>
#include <adam_optimizer.h>

#ifndef LAYER_H
#define LAYER_H

using namespace std;

class Layer
{
	Activation activation;
	int in_count;
	int out_count;

	vec2d weights;
	vec1d biases;

	vec1d m_t;
	vec1d v_t;

	vec1d inputs;
	vec1d errors;
	vec1d args;
	vec1d outputs;
public:
	Layer() {}

	Layer(Activation activation, int in_count, int out_count, vec2d weights, vec1d biases)
	{
		this->activation = activation;
		this->in_count = in_count;
		this->out_count = out_count;

		this->weights = weights;
		this->biases = biases;

		m_t = vec1d(out_count);
		v_t = vec1d(out_count);

		inputs = vec1d(in_count);
		errors = vec1d(out_count);
		args = vec1d(out_count);
		outputs = vec1d(out_count);
	}

	vec1d forward(vec1d inputs)
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

	vec1d backward(vec1d errors_next)
	{
		errors.make_zero();

		#pragma omp parallel for
		for(int i = 0; i < out_count; ++i)
		{
			errors[i] = activation.der(outputs[i], args[i]) * errors_next[i];
		}

		vec1d errors_back(in_count);

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

	void fit(int t, AdamOptimizer& adam)
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