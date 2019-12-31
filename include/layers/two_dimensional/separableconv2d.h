#include <map>
#include <omp.h>  // OpenMP headers
#include <ops.h>
#include <types.h>
#include <layers/two_dimensional/layer2d.h>

#ifndef SEPARABLE_CONV2D_H
#define SEPARABLE_CONV2D_H

using namespace std;

class SeparableConv2D : public Layer2D
{
	tensor_4d depthwise_kernel;
	tensor_4d pointwise_kernel;

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

	tensor_4d dMt;  // depthwise M_t
	tensor_4d dVt;  // depthwise V_t

	tensor_4d pMt;  // pointwise M_t
	tensor_4d pVt;  // pointwise V_t

	tensor_3d inputs;
	tensor_3d gradients;
	tensor_3d intermediate;
	tensor_3d inter_gradients;
	tensor_3d gradient_back;
	tensor_3d outputs;
public:
	SeparableConv2D() {}

	SeparableConv2D(tensor_4d depthwise_kernel, tensor_4d pointwise_kernel, map<string, int> params)
	{
		this->depthwise_kernel = depthwise_kernel;
		this->pointwise_kernel = pointwise_kernel;

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

		inputs = tensor_3d(input_count, tensor_2d(input_width, tensor_1d(input_height)));
		intermediate = tensor_3d(input_count, tensor_2d(out_width, tensor_1d(out_height)));
		gradients = tensor_3d(kernel_count, tensor_2d(out_width, tensor_1d(out_height)));
		inter_gradients = tensor_3d(input_count, tensor_2d(out_width, tensor_1d(out_height)));
		gradient_back = tensor_3d(input_count, tensor_2d(input_width, tensor_1d(input_height)));
		outputs = tensor_3d(kernel_count, tensor_2d(out_width, tensor_1d(out_height)));

		dMt = tensor_4d(input_count, tensor_3d(1, tensor_2d(kernel_width, tensor_1d(kernel_height))));
		dVt = tensor_4d(input_count, tensor_3d(1, tensor_2d(kernel_width, tensor_1d(kernel_height))));

		pMt = tensor_4d(kernel_count, tensor_3d(input_count, tensor_2d(1, tensor_1d(1))));
		pVt = tensor_4d(kernel_count, tensor_3d(input_count, tensor_2d(1, tensor_1d(1))));
	}

	tensor_3d forward(tensor_3d feature_map)
	{
		inputs = feature_map;
		make_zero(intermediate);
		make_zero(outputs);

		// Depthwise convolution: input_count x 1 x kernel_width x kernel_height.
		for(int i = 0; i < input_count; ++i)
		{
			conv2d(feature_map[i], depthwise_kernel[i][0], intermediate[i]);
		}

		// Pointwise convolution: kernel_count x input_count x 1 x 1.
		for(int i = 0; i < kernel_count; ++i)
		{
			for(int j = 0; j < input_count; ++j)
			{
				conv2d(intermediate[j], pointwise_kernel[i][j], outputs[i]);
			}
		}

		return outputs;
	}

	tensor_3d backward(tensor_3d gradients_next)
	{
		gradients = gradients_next;
		make_zero(inter_gradients);
		make_zero(gradient_back);

		// Pointwise convolution's backpropagation
		for(int i = 0; i < input_count; ++i)
		{
			for(int j = 0; j < kernel_count; ++j)
			{
				conv2d(gradients[j], rot180(pointwise_kernel[j][i]), inter_gradients[i]);
			}
		}

		// Depthwise convolution's backpropagation
		for(int i = 0; i < input_count; ++i)
		{
			conv2d(inter_gradients[i], rot180(depthwise_kernel[i][0]), gradient_back[i]);
		}

		return gradient_back;
	}

	void fit(int t, AdamOptimizer& adam)
	{
		// Pointwise kernel fitting
		for(int i = 0; i < kernel_count; ++i)
		{
			for(int j = 0; j < input_count; ++j)
			{
				tensor_2d gradient_now(1, tensor_1d(1));
				conv2d(gradients[i], inputs[j], gradient_now);
				gradient_now = rot180(gradient_now);

				double update = adam.optimize(t, pMt[i][j][0][0], pVt[i][j][0][0], gradient_now[0][0]);
				pointwise_kernel[i][j][0][0] += update;
			}
		}

		// Depthwise kernel fitting
		for(int i = 0; i < input_count; ++i)
		{
			tensor_2d gradient_now(kernel_width, tensor_1d(kernel_height));
			conv2d(inter_gradients[i], inputs[i], gradient_now);
			gradient_now = rot180(gradient_now);

			#pragma omp parallel for
			for(int x = 0; x < kernel_width; ++x)
			{
				for(int y = 0; y < kernel_height; ++y)
				{
					double update = adam.optimize(t, dMt[i][0][x][y], dVt[i][0][x][y], gradient_now[x][y]);
					depthwise_kernel[i][0][x][y] += update;
				}
			}
		}
	}
};

#endif