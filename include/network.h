#include <vector>
#include <activation.h>
#include <adam_optimizer.h>
#include <layer.h>

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
	vector<Layer> layers;

	vec1d inputs;
	vec1d outputs;
public:
	Network(int in_count, int out_count, vector<LayerDescription> layer_descrs)
	{
		this->in_count = in_count;
		this->out_count = out_count;

		layer_len = layer_descrs.size();

		layers = vector<Layer>();
		int layer_in = in_count;

		for(int i = 0; i < layer_len; ++i)
		{
			int layer_out = layer_descrs[i].length;

			vec2d weights(layer_out, layer_in);
			weights.make_random();

			vec1d biases(layer_out);
			biases.make_random();

			Layer layer(layer_descrs[i].activation, layer_in, layer_out, weights, biases);
			layers.push_back(layer);

			layer_in = layer_out;
		}
	}

	vec1d forward(vec1d inputs)
	{
		this->inputs = inputs;
		vec1d intermediate = inputs;

		for(int i = 0; i < layer_len; ++i)
		{
			intermediate = layers[i].forward(intermediate);
		}

		this->outputs = intermediate;
		return this->outputs;
	}

	vec1d backward(vec1d outputs_true)
	{
		vec1d errors(out_count);

		for(int i = 0; i < out_count; ++i)
		{
			errors[i] = this->outputs[i] - outputs_true[i];
		}

		for(int i = layer_len - 1; i >= 0; --i)
		{
			errors = layers[i].backward(errors);
		}

		return errors;
	}

	void fit(int t, AdamOptimizer& adam)
	{
		for(int i = 0; i < layer_len; ++i)
		{
			layers[i].fit(t, adam);
		}
	}
};

#endif