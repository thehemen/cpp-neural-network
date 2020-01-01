#include <cmath>
#include <climits>
#include <omp.h>  // OpenMP headers
#include <layers/one_dimensional/layer1d.h>
#include <ops.h>
#include <types.h>

#ifndef SOFTMAX_H
#define SOFTMAX_H

using namespace std;

class Softmax : public Layer1D
{
	int out_count;
public:
	Softmax(map<string, int> params)
	{
		out_count = params["out_count"];
	}

	tensor_1d forward(tensor_1d inputs) override
	{
		tensor_1d outputs(out_count);
		double max_value = -DBL_MAX;

		for(int i = 0; i < out_count; ++i)
		{
			if(inputs[i] > max_value)
			{
				max_value = inputs[i];
			}
		}

		double exp_sum = 0.0;

		for(int i = 0; i < out_count; ++i)
		{
			outputs[i] = exp(inputs[i] - max_value);
			exp_sum = exp_sum + outputs[i];
		}

		#pragma omp parallel for
		for(int i = 0; i < out_count; ++i)
		{
			outputs[i] = outputs[i] / exp_sum;
		}

		return outputs;
	};

	tensor_1d backward(tensor_1d gradients) override
	{
		/*
			The gradient of Softmax:
			{gradient_i} = {predicted_i} - {true_i}
			that is calculated by "Network" earlier.
		*/
		return gradients;
	};

	void fit(int t, AdamOptimizer& adam) override {}
};

#endif