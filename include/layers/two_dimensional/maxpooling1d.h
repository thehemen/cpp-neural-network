#include <map>
#include <omp.h>  // OpenMP headers
#include <ops.h>
#include <types.h>
#include <layers/two_dimensional/layer2d.h>

#ifndef MAXPOOLING_1D_H
#define MAXPOOLING_1D_H

using namespace std;

class MaxPooling1D : public Layer2D
{
	int width;

	int input_length;
	int input_width;
	int out_width;

	tensor_2d mask;
	tensor_2d outputs;
public:
	MaxPooling1D() {}

	MaxPooling1D(map<string, int> params)
	{
		width = params["width"];
		input_length = params["input_count"];
		input_width = params["input_width"];
		out_width = params["out_width"];

		mask = tensor_2d(input_length, tensor_1d(input_width));
		outputs = tensor_2d(input_length, tensor_1d(out_width));
	}

	tensor_2d forward(tensor_2d inputs) override
	{
		make_zero(mask);
		make_zero(outputs);

		#pragma omp parallel for
		for(int i = 0; i < input_length; ++i)
		{
			for(int x = 0; x < out_width; ++x)
		    {
	        	int x_max = 0;
	        	double value_max = 0.0;

	            for(int k_x = 0; k_x < width; ++k_x)
	            {
                	int x_now = x * width + k_x;
                	double value_now = inputs[i][x_now];

                	if(value_now > value_max)
                	{
                		x_max = x_now;
                		value_max = value_now;
                	}
	            }

	            outputs[i][x] = value_max;
	            mask[i][x_max] = 1.0;
		    }
		}

	    return outputs;
	}

	tensor_2d backward(tensor_2d gradients) override
	{
		tensor_2d gradient_back(input_length, tensor_1d(input_width));

		#pragma omp parallel for
		for(int i = 0; i < input_length; ++i)
		{
			for(int x = 0; x < out_width; ++x)
		    {
	            for(int k_x = 0; k_x < width; ++k_x)
	            {
                	int x_now = x * width + k_x;

                	if(mask[i][x_now])
                	{
                		gradient_back[i][x_now] = gradients[i][x];
                	}
	            }
		    }
		}

	    return gradient_back;
	}

	void fit(int t, AdamOptimizer& adam) override {}
};

#endif