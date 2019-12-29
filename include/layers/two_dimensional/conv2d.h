#include <map>
#include <omp.h>  // OpenMP headers
#include <ops.h>
#include <types.h>
#include <layers/two_dimensional/layer2d.h>

#ifndef CONV2D_H
#define CONV2D_H

using namespace std;

class Conv2D : public Layer2D
{
	tensor_4d kernel;
	tensor_1d bias;

	int kernel_count;
	int kernel_width;
	int kernel_height;

	int input_count;
	int input_width;
	int input_height;

	int out_width;
	int out_height;

	int padding_width;
	int padding_height;

	tensor_4d m_t;
	tensor_4d v_t;

	tensor_3d inputs;
	tensor_3d gradients;
	tensor_3d outputs;
public:
	Conv2D() {}

	Conv2D(tensor_4d kernel, tensor_1d bias, map<string, int> params)
	{
		this->kernel = kernel;
		this->bias = bias;

		kernel_count = params["count"];
		kernel_width = params["width"];
		kernel_height = params["height"];

		input_count = params["input_count"];
		input_width = params["input_width"];
		input_height = params["input_height"];

		out_width = params["out_width"];
		out_height = params["out_height"];

		padding_width = params["padding_width"];
		padding_height = params["padding_height"];

		m_t = tensor_4d(kernel_count, tensor_3d(input_count, tensor_2d(kernel_width, tensor_1d(kernel_height))));
		v_t = tensor_4d(kernel_count, tensor_3d(input_count, tensor_2d(kernel_width, tensor_1d(kernel_height))));
	}

	tensor_3d forward(tensor_3d feature_map)
	{
		inputs = feature_map;
		outputs = tensor_3d(kernel_count, tensor_2d(out_width, tensor_1d(out_height)));

		for(int i = 0; i < kernel_count; ++i)
		{
			for(int j = 0; j < input_count; ++j)
			{
				conv2d(feature_map[j], kernel[i][j], outputs[i], bias[i]);
			}
		}

		return outputs;
	}

	tensor_3d backward(tensor_3d gradients_next)
	{
		gradients = tensor_3d(kernel_count, tensor_2d(out_width, tensor_1d(out_height)));

		for(int i = 0; i < kernel_count; ++i)
		{
			gradients[i] = zeropadding2d(gradients[i], padding_width, padding_height);
		}

		tensor_3d gradient_back(input_count, tensor_2d(input_width, tensor_1d(input_height)));

		for(int i = 0; i < input_count; ++i)
		{
			for(int j = 0; j < kernel_count; ++j)
			{
				conv2d(gradients[j], rot180(kernel[j][i]), gradient_back[i]);
			}
		}

		return gradient_back;
	}

	void fit(int t, AdamOptimizer& adam)
	{
		for(int i = 0; i < kernel_count; ++i)
		{
			for(int j = 0; j < input_count; ++j)
			{
				tensor_2d gradient_now(kernel_width, tensor_1d(kernel_height));
				conv2d(gradients[i], inputs[j], gradient_now);
				gradient_now = rot180(gradient_now);

				#pragma omp parallel for
				for(int x = 0; x < kernel_width; ++x)
				{
					for(int y = 0; y < kernel_height; ++y)
					{
						double update = adam.optimize(t, m_t[i][j][x][y], v_t[i][j][x][y], gradient_now[x][y]);
						kernel[i][j][x][y] += update;
						bias[i] += update;
					}
				}
			}
		}
	}
};

#endif