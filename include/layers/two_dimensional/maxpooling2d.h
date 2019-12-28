#include <map>
#include <omp.h>  // OpenMP headers
#include <ops.h>
#include <types.h>
#include <layers/two_dimensional/layer2d.h>

#ifndef MAXPOOLING_2D_H
#define MAXPOOLING_2D_H

using namespace std;

class MaxPooling2D : public Layer2D
{
	int width;
	int height;

	int input_count;
	int input_width;
	int input_height;

	int out_width;
	int out_height;

	tensor_3d mask;
	tensor_3d outputs;
	tensor_3d gradient_back;
public:
	MaxPooling2D() {}

	MaxPooling2D(map<string, int> params)
	{
		width = params["width"];
		height = params["height"];

		input_count = params["input_count"];
		input_width = params["input_width"];
		input_height = params["input_height"];

		out_width = params["out_width"];
		out_height = params["out_height"];

		mask = tensor_3d(input_count, tensor_2d(input_width, tensor_1d(input_height)));
		outputs = tensor_3d(input_count, tensor_2d(out_width, tensor_1d(out_height)));
		gradient_back = tensor_3d(input_count, tensor_2d(input_width, tensor_1d(input_height)));
	}

	tensor_3d forward(tensor_3d feature_map) override
	{
		make_zero(mask);
		make_zero(outputs);

		#pragma omp parallel for
		for(int i = 0; i < input_count; ++i)
		{
			for(int x = 0; x < out_width; ++x)
		    {
		        for(int y = 0; y < out_height; ++y)
		        {
		        	int x_max = 0;
		        	int y_max = 0;
		        	double value_max = 0.0;

		            for(int k_x = 0; k_x < width; ++k_x)
		            {
		                for(int k_y = 0; k_y < height; ++k_y)
		                {
		                	int x_now = x * width + k_x;
		                	int y_now = y * height + k_y;
		                	double value_now = feature_map[i][x_now][y_now];

		                	if(value_now > value_max)
		                	{
		                		x_max = x_now;
		                		y_max = y_now;
		                		value_max = value_now;
		                	}
		                }
		            }

		            outputs[i][x][y] = value_max;
		            mask[i][x_max][y_max] = 1.0;
		        }
		    }
		}

	    return outputs;
	}

	tensor_3d backward(tensor_3d gradients) override
	{
		make_zero(gradient_back);

		#pragma omp parallel for
		for(int i = 0; i < input_count; ++i)
		{
			for(int x = 0; x < out_width; ++x)
		    {
		        for(int y = 0; y < out_height; ++y)
		        {
		            for(int k_x = 0; k_x < width; ++k_x)
		            {
		                for(int k_y = 0; k_y < height; ++k_y)
		                {
		                	int x_now = x * width + k_x;
		                	int y_now = y * height + k_y;

		                	if(mask[i][x_now][y_now])
		                	{
		                		gradient_back[i][x_now][y_now] = gradients[i][x][y];
		                	}
		                }
		            }
		        }
		    }
		}

	    return gradient_back;
	}

	void fit(int t, AdamOptimizer& adam) override {}
};

#endif