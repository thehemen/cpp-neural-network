#include <vector>
#include <activation.h>
#include <adam_optimizer.h>
#include <tensors/tensor_1d.h>
#include <tensors/tensor_2d.h>
#include <layers/one_dimensional/layer1d.h>
#include <layers/one_dimensional/dense.h>

#ifndef NETWORK_H
#define NETWORK_H

using namespace std;

struct LayerDescription
{
	Activation activation;
	int length;

	LayerDescription(Activation activation, int length)
	{
		this->activation = activation;
		this->length = length;
	}
};

class Network
{
	int in_count;
	int layer_len;
	int out_count;
	
	vector<Layer1D*> layers;

	tensor_1d inputs;
	tensor_1d outputs;
public:
	Network(int in_count, int out_count, vector<LayerDescription> layer_descrs)
	{
		this->in_count = in_count;
		this->out_count = out_count;

		layer_len = layer_descrs.size();

		layers = vector<Layer1D*>();
		int layer_in = in_count;

		for(int i = 0; i < layer_len; ++i)
		{
			int layer_out = layer_descrs[i].length;

			tensor_2d weights(layer_out, layer_in);
			weights.make_random();

			tensor_1d biases(layer_out);
			biases.make_random();

			Dense* layer = new Dense(layer_descrs[i].activation, layer_in, layer_out, weights, biases);
			layers.push_back(layer);

			layer_in = layer_out;
		}
	}

	tensor_1d forward(tensor_1d inputs)
	{
		this->inputs = inputs;
		tensor_1d intermediate = inputs;

		for(int i = 0; i < layer_len; ++i)
		{
			intermediate = layers[i]->forward(intermediate);
		}

		this->outputs = intermediate;
		return this->outputs;
	}

	tensor_1d backward(tensor_1d outputs_true)
	{
		tensor_1d errors(out_count);

		for(int i = 0; i < out_count; ++i)
		{
			errors[i] = this->outputs[i] - outputs_true[i];
		}

		for(int i = layer_len - 1; i >= 0; --i)
		{
			errors = layers[i]->backward(errors);
		}

		return errors;
	}

	void fit(int t, AdamOptimizer& adam)
	{
		for(int i = 0; i < layer_len; ++i)
		{
			layers[i]->fit(t, adam);
		}
	}
};

#endif