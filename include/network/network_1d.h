#include <vector>
#include <sstream>
#include <map>

#include <adam_optimizer.h>
#include <types.h>

#include <layers/one_dimensional/layer1d.h>

#ifndef NETWORK_1D_H
#define NETWORK_1D_H

using namespace std;

class Network1D
{
	vector<Layer1D*> layer1d_s;
	int layer1d_len;
	tensor_1d outputs;
public:
	Network1D(vector<Layer1D*> layer1d_s)
	{
		this->layer1d_s = vector<Layer1D*>(layer1d_s);
		layer1d_len = layer1d_s.size();
	}

	tensor_1d forward(tensor_1d inputs)
	{
		tensor_1d tensor1d = inputs;

		for(int i = 0; i < layer1d_len; ++i)
		{
			tensor1d = layer1d_s[i]->forward(tensor1d);
		}

		outputs = tensor1d;
		return outputs;
	}

	tensor_1d backward(tensor_1d outputs_true)
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
		for(int i = 0; i < layer1d_len; ++i)
		{
			layer1d_s[i]->fit(t, adam);
		}
	}
};

#endif