#include <map>
#include <omp.h>  // OpenMP headers
#include <ops.h>
#include <types.h>
#include <layers/three_dimensional/layer3d.h>

#ifndef ACTIVATION_3D_H
#define ACTIVATION_3D_H

using namespace std;

class Activation3D : public Layer3D
{
	Activation activation;

	int out_count;
	int out_width;
	int out_height;

	tensor_3d inputs;
	tensor_3d outputs;
public:
	Activation3D(Activation activation, map<string, int> params)
	{
		this->activation = activation;

		out_count = params["out_count"];
		out_width = params["out_width"];
		out_height = params["out_height"];

		inputs = tensor_3d(out_count, tensor_2d(out_width, tensor_1d(out_height)));
		outputs = tensor_3d(out_count, tensor_2d(out_width, tensor_1d(out_height)));
	}

	tensor_3d forward(tensor_3d feature_map) override
	{
		inputs = feature_map;
		make_zero(outputs);

		#pragma omp parallel for
		for(int i = 0; i < out_count; ++i)
		{
			for(int x = 0; x < out_width; ++x)
			{
				for(int y = 0; y < out_height; ++y)
				{
					outputs[i][x][y] = activation.get(inputs[i][x][y]);
				}
			}
		}

		return outputs;
	};

	tensor_3d backward(tensor_3d gradients) override
	{
		tensor_3d gradients_back = tensor_3d(out_count, tensor_2d(out_width, tensor_1d(out_height)));

		#pragma omp parallel for
		for(int i = 0; i < out_count; ++i)
		{
			for(int x = 0; x < out_width; ++x)
			{
				for(int y = 0; y < out_height; ++y)
				{
					gradients_back[i][x][y] = activation.der(outputs[i][x][y], inputs[i][x][y]) * gradients[i][x][y];
				}
			}
		}

		return gradients_back;
	};

	void fit(int t, AdamOptimizer& adam) override {};
};

#endif