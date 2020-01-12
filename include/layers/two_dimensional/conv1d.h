#include <map>
#include <omp.h>  // OpenMP headers
#include <ops.h>
#include <types.h>
#include <layers/two_dimensional/layer2d.h>

#ifndef CONV1D_H
#define CONV1D_H

using namespace std;

class Conv1D : public Layer2D
{
	tensor_3d kernel;

	int kernel_length;
	int kernel_width;

	int input_length;
	int input_width;

	int out_width;
	int padding_width;

	tensor_3d Mt;
	tensor_3d Vt;

	tensor_2d inputs;
	tensor_2d gradients;
	tensor_2d outputs;
public:
	Conv1D() {}

	Conv1D(tensor_3d kernel, map<string, int> params)
	{
		this->kernel = kernel;

		kernel_length = params["count"];
		kernel_width = params["width"];

		input_length = params["input_count"];
		input_width = params["input_width"];

		out_width = params["out_width"];
		padding_width = params["padding_width"];

		Mt = tensor_3d(kernel_length, tensor_2d(input_length, tensor_1d(kernel_width)));
		Vt = tensor_3d(kernel_length, tensor_2d(input_length, tensor_1d(kernel_width)));
	}

	tensor_2d forward(tensor_2d inputs)
	{
		this->inputs = inputs;
		outputs = tensor_2d(kernel_length, tensor_1d(out_width));

		for(int i = 0; i < kernel_length; ++i)
		{
			for(int j = 0; j < input_length; ++j)
			{
				conv1d(this->inputs[j], kernel[i][j], outputs[i]);
			}
		}

		return outputs;
	}

	tensor_2d backward(tensor_2d gradients)
	{
		this->gradients = gradients;

		for(int i = 0; i < kernel_length; ++i)
		{
			this->gradients[i] = zeropadding1d(this->gradients[i], padding_width);
		}

		tensor_2d gradient_back(input_length, tensor_1d(input_width));

		for(int i = 0; i < input_length; ++i)
		{
			for(int j = 0; j < kernel_length; ++j)
			{
				conv1d(this->gradients[j], rot180_1d(kernel[j][i]), gradient_back[i]);
			}
		}

		return gradient_back;
	}

	void fit(int t, AdamOptimizer& adam)
	{
		for(int i = 0; i < kernel_length; ++i)
		{
			for(int j = 0; j < input_length; ++j)
			{
				tensor_1d gradient_now(kernel_width);
				conv1d(gradients[i], inputs[j], gradient_now);
				gradient_now = rot180_1d(gradient_now);

				#pragma omp parallel for
				for(int x = 0; x < kernel_width; ++x)
				{
					double update = adam.optimize(t, Mt[i][j][x], Vt[i][j][x], gradient_now[x]);
					kernel[i][j][x] += update;
				}
			}
		}
	}
};

#endif