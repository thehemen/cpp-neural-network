#include <vector>
#include <sstream>
#include <map>

#include <adam_optimizer.h>
#include <types.h>

#include <layers/one_dimensional/layer1d.h>
#include <layers/two_to_one_dim/layer2to1d.h>
#include <layers/two_dimensional/layer2d.h>

#ifndef NETWORK_H
#define NETWORK_H

using namespace std;

class Network
{
	vector<Layer2D*> layer2d_s;
	Layer2to1D* layer2to1d;
	vector<Layer1D*> layer1d_s;

	int layer2d_len;
	int layer1d_len;

	tensor_1d outputs;
public:
	Network(vector<Layer2D*> layer2d_s, Layer2to1D* layer2to1d, vector<Layer1D*> layer1d_s)
	{
		this->layer2d_s = vector<Layer2D*>(layer2d_s);
		this->layer2to1d = layer2to1d;
		this->layer1d_s = vector<Layer1D*>(layer1d_s);

		layer2d_len = layer2d_s.size();
		layer1d_len = layer1d_s.size();
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

		outputs = tensor1d;
		return outputs;
	}

	tensor_1d forward_1to1d(tensor_1d inputs)
	{
		tensor_1d tensor1d = inputs;

		for(int i = 0; i < layer1d_len; ++i)
		{
			tensor1d = layer1d_s[i]->forward(tensor1d);
		}

		outputs = tensor1d;
		return outputs;
	}

	tensor_3d backward_1to2d(tensor_1d outputs_true)
	{
		int out_count = outputs.size();
		tensor_1d tensor1d(out_count);

		for(int i = 0; i < out_count; ++i)
		{
			tensor1d[i] = outputs[i] - outputs_true[i];
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
		int out_count = outputs.size();
		tensor_1d tensor1d(out_count);

		for(int i = 0; i < out_count; ++i)
		{
			tensor1d[i] = outputs[i] - outputs_true[i];
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