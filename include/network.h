#include <vector>
#include <sstream>
#include <map>

#include <activation.h>
#include <adam_optimizer.h>
#include <types.h>

#include <layers/one_dimensional/layer1d.h>
#include <layers/one_dimensional/dense.h>

#include <layers/two_to_one_dim/layer2to1d.h>
#include <layers/two_to_one_dim/flatten.h>

#include <layers/two_dimensional/layer2d.h>
#include <layers/two_dimensional/conv2d.h>
#include <layers/two_dimensional/maxpooling2d.h>

#ifndef NETWORK_H
#define NETWORK_H

using namespace std;

struct LayerDescription
{
	string layerType;
	map<string, int> params;
	Activation activation;

	LayerDescription(string layerType, map<string, int> params = map<string, int>(), Activation activation = Activation(ActivationType::NONE))
	{
		this->layerType = layerType;
		this->params = params;
		this->activation = activation;
	}
};

class Network
{
	int input_width;
	int input_height;

	vector<Layer2D*> layer2d_s;
	int layer2d_len;

	Layer2to1D* layer2to1d;

	vector<Layer1D*> layer1d_s;
	int layer1d_len;

	string info;
	tensor_1d outputs;
public:
	Network(vector<LayerDescription> layer_descrs, map<string, int> params)
	{
		layer2d_s = vector<Layer2D*>();
		layer1d_s = vector<Layer1D*>();

		stringstream ostream;
		int input_count = 1;
		int input_width = 1;
		int input_height = 1;

		if(layer_descrs[0].layerType != "dense")
		{
			input_width = params["width"];
			input_height = params["height"];
			ostream << input_width << "x" << input_height << "\n";
		}
		else
		{
			input_count = params["count"];
			ostream << input_count << "\n";
		}

		for(int i = 0, layer_len = layer_descrs.size(); i < layer_len; ++i)
		{
			string layerType = layer_descrs[i].layerType;
			map<string, int> params = layer_descrs[i].params;
			Activation activation = layer_descrs[i].activation;

			if(layerType == "conv2d")
			{
				int count = params["count"];
				int width = params["width"];
				int height = params["height"];

				tensor_4d kernel(count, tensor_3d(input_count, tensor_2d(width, tensor_1d(height))));
				make_random(kernel);

				tensor_1d biases(count);
				make_random(biases);

				params["input_count"] = input_count;
				params["input_width"] = input_width;
				params["input_height"] = input_height;

				int out_count = count;
				int out_width = input_width - width + 1;
				int out_height = input_height - height + 1;

				params["out_width"] = out_width;
				params["out_height"] = out_height;

				params["padding_width"] = input_width - out_width;
				params["padding_height"] = input_height - out_height;

				Conv2D* conv2d = new Conv2D(activation, kernel, biases, params);
				layer2d_s.push_back(conv2d);

				input_count = out_count;
				input_width = out_width;
				input_height = out_height;

				ostream << input_count << "x" << input_width << "x" << input_height;
				ostream << "\tConv2D\n";
			}
			else if(layerType == "maxpooling2d")
			{
				int width = params["width"];
				int height = params["height"];

				int out_width = input_width / width;
				int out_height = input_height / height;

				params["input_count"] = input_count;
				params["input_width"] = input_width;
				params["input_height"] = input_height;

				params["out_width"] = out_width;
				params["out_height"] = out_height;

				MaxPooling2D* maxpool2d = new MaxPooling2D(params);
				layer2d_s.push_back(maxpool2d);

				input_width = out_width;
				input_height = out_height;

				ostream << input_count << "x" << input_width << "x" << input_height;
				ostream << "\tMaxPooling2D\n";
			}
			else if(layerType == "flatten")
			{
				params["count"] = input_count;
				params["width"] = input_width;
				params["height"] = input_height;

				Flatten* flatten = new Flatten(params);
				layer2to1d = flatten;

				input_count *= input_width * input_height;
				input_width = 1;
				input_height = 1;

				ostream << input_count;
				ostream << "\tFlatten\n";
			}
			else if(layerType == "dense")
			{
				int dense_out = params["length"];

				tensor_2d weights(dense_out, tensor_1d(input_count));
				make_random(weights);

				tensor_1d biases(dense_out);
				make_random(biases);

				params["input_count"] = input_count;

				Dense* dense = new Dense(activation, weights, biases, params);
				layer1d_s.push_back(dense);

				input_count = dense_out;

				ostream << input_count;
				ostream << "\tDense\n";
			}
		}

		layer2d_len = layer2d_s.size();
		layer1d_len = layer1d_s.size();
		info = ostream.str();
	}

	string get_shapes()
	{
		return info;
	}

	tensor_1d forward_2to1d(tensor_3d inputs)
	{
		tensor_3d tensor3d = inputs;

		for(int i = 0; i < layer2d_len; ++i)
		{
			tensor3d = layer2d_s[i]->forward(tensor3d);
		}

		tensor_1d tensor1d = layer2to1d->forward(tensor3d);

		for(int i = 0; i < layer1d_len; ++i)
		{
			tensor1d = layer1d_s[i]->forward(tensor1d);
		}

		this->outputs = tensor1d;
		return this->outputs;
	}

	tensor_1d forward_1to1d(tensor_1d inputs)
	{
		tensor_1d tensor1d = inputs;

		for(int i = 0; i < layer1d_len; ++i)
		{
			tensor1d = layer1d_s[i]->forward(tensor1d);
		}

		this->outputs = tensor1d;
		return this->outputs;
	}

	tensor_3d backward_1to2d(tensor_1d outputs_true)
	{
		int out_count = this->outputs.size();
		tensor_1d tensor1d(out_count);

		for(int i = 0; i < out_count; ++i)
		{
			tensor1d[i] = this->outputs[i] - outputs_true[i];
		}

		for(int i = layer1d_len - 1; i >= 0; --i)
		{
			tensor1d = layer1d_s[i]->backward(tensor1d);
		}

		tensor_3d tensor3d = layer2to1d->backward(tensor1d);

		for(int i = layer2d_len - 1; i >= 0; --i)
		{
			tensor3d = layer2d_s[i]->backward(tensor3d);
		}

		return tensor3d;
	}

	tensor_1d backward_1to1d(tensor_1d outputs_true)
	{
		int out_count = this->outputs.size();
		tensor_1d tensor1d(out_count);

		for(int i = 0; i < out_count; ++i)
		{
			tensor1d[i] = this->outputs[i] - outputs_true[i];
		}

		for(int i = layer1d_len - 1; i >= 0; --i)
		{
			tensor1d = layer1d_s[i]->backward(tensor1d);
		}

		return tensor1d;
	}

	void fit(int t, AdamOptimizer& adam)
	{
		for(int i = 0; i < layer2d_len; ++i)
		{
			layer2d_s[i]->fit(t, adam);
		}

		for(int i = 0; i < layer1d_len; ++i)
		{
			layer1d_s[i]->fit(t, adam);
		}
	}
};

#endif