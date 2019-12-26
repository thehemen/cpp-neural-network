#include <map>
#include <omp.h>  // OpenMP headers
#include <ops.h>
#include <tensors/tensor_2d.h>
#include <tensors/tensor_3d.h>
#include <layers/two_dimensional/layer2d.h>

#ifndef CONV2D_H
#define CONV2D_H

#include <iostream>

using namespace std;

class Conv2D : public Layer2D
{
	Activation activation;
	tensor_3d kernel;
	tensor_1d bias;

	int kernel_count;
	int kernel_width;
	int kernel_height;

	int input_count;
	int input_width;
	int input_height;

	int out_count;
	int out_width;
	int out_height;

	int padding_width;
	int padding_height;

	int gradient_width;
	int gradient_height;

	tensor_3d m_t;
	tensor_3d v_t;

	tensor_3d inputs;
	tensor_3d gradients;
	tensor_3d args;
	tensor_3d outputs;
public:
	Conv2D() {}

	Conv2D(Activation activation, tensor_3d kernel, tensor_1d bias, map<string, int> params)
	{
		this->activation = activation;
		this->kernel = kernel;
		this->bias = bias;

		kernel_count = params["count"];
		kernel_width = params["width"];
		kernel_height = params["height"];

		input_count = params["input_count"];
		input_width = params["input_width"];
		input_height = params["input_height"];

		out_count = params["out_count"];
		out_width = params["out_width"];
		out_height = params["out_height"];

		padding_width = params["padding_width"];
		padding_height = params["padding_height"];

		gradient_width = params["gradient_width"];
		gradient_height = params["gradient_height"];

		m_t = tensor_3d(kernel_count, kernel_width, kernel_height);
		v_t = tensor_3d(kernel_count, kernel_width, kernel_height);
	}

	tensor_3d forward(tensor_3d feature_map)
	{
		inputs = feature_map;
		vector<tensor_2d> output_vec;

		for(int i = 0; i < kernel_count; ++i)
		{
			for(int j = 0; j < input_count; ++j)
			{
				tensor_2d output_now = conv2d(feature_map[j], kernel[i]);
				output_vec.push_back(output_now);
			}
		}

	    outputs = tensor_3d(output_vec);
		args = tensor_3d(out_count, out_width, out_height);

		#pragma omp parallel for
		for(int i = 0; i < out_count; ++i)
		{
			for(int x = 0; x < out_width; ++x)
			{
				for(int y = 0; y < out_height; ++y)
				{
					double argument = outputs[i][x][y] + bias[i % input_count];
					args[i][x][y] = argument;
					outputs[i][x][y] = activation.get(argument);
				}
			}
		}

		return outputs;
	}

	tensor_3d backward(tensor_3d gradients_next)
	{
		gradients = tensor_3d(out_count, out_width, out_height);
		vector<tensor_2d> gradient_back_vec;

		for(int i = 0; i < input_count; ++i)
		{
			tensor_2d gradient_back_sum(input_width, input_height);

			for(int j = 0; j < kernel_count; ++j)
			{
				int index = i + j * input_count;

				#pragma omp parallel for
				for(int x = 0; x < out_width; ++x)
				{
					for(int y = 0; y < out_height; ++y)
					{
						gradients[index][x][y] = activation.der(outputs[index][x][y], args[index][x][y]) * gradients_next[index][x][y];
					}
				}

				gradients[index] = zeropadding2d(gradients[index], padding_width, padding_height);
				tensor_2d gradient_back  = conv2d(gradients[index], rot180(kernel[j]));
				
				#pragma omp parallel for
				for(int x = 0; x < input_width; ++x)
				{
					for(int y = 0; y < input_height; ++y)
					{
						gradient_back_sum[x][y] += gradient_back[x][y];
					}
				}
			}

			gradient_back_vec.push_back(gradient_back_sum);
		}

		tensor_3d gradients_back(gradient_back_vec);
		return gradients_back;
	}

	void fit(int t, AdamOptimizer& adam)
	{
		for(int i = 0; i < kernel_count; ++i)
		{
			tensor_2d gradient_sum(gradient_width, gradient_height);

			for(int j = 0; j < input_count; ++j)
			{
				#pragma omp parallel for
				for(int x = 0; x < gradient_width; ++x)
				{
					for(int y = 0; y < gradient_height; ++y)
					{
						gradient_sum[x][y] += gradients[i * input_count + j][x][y];
					}
				}

				tensor_2d gradient_now = rot180(conv2d(gradient_sum, inputs[j]));

				#pragma omp parallel for
				for(int x = 0; x < kernel_width; ++x)
				{
					for(int y = 0; y < kernel_height; ++y)
					{
						double update = adam.optimize(t, m_t[i][x][y], v_t[i][x][y], gradient_now[x][y]);
						kernel[i][x][y] += update;
						bias[i] += update;
					}
				}
			}
		}
	}
};

#endif