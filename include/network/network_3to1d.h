#include <vector>
#include <sstream>
#include <map>

#include <adam_optimizer.h>
#include <types.h>

#include <layers/one_dimensional/layer1d.h>
#include <layers/three_to_one_dim/layer3to1d.h>
#include <layers/three_dimensional/layer3d.h>

#ifndef NETWORK_3to1D_H
#define NETWORK_3to1D_H

using namespace std;

class Network3to1D
{
	vector<Layer3D*> layer3d_s;
	Layer3to1D* layer3to1d;
	vector<Layer1D*> layer1d_s;

	int layer3d_len;
	int layer1d_len;

	tensor_1d outputs;
public:
	Network3to1D(vector<Layer3D*> layer3d_s, Layer3to1D* layer3to1d, vector<Layer1D*> layer1d_s)
	{
		this->layer3d_s = vector<Layer3D*>(layer3d_s);
		this->layer3to1d = layer3to1d;
		this->layer1d_s = vector<Layer1D*>(layer1d_s);

		layer3d_len = layer3d_s.size();
		layer1d_len = layer1d_s.size();
	}

	tensor_1d forward(tensor_3d inputs)
	{
		tensor_3d tensor3d = inputs;

		for(int i = 0; i < layer3d_len; ++i)
		{
			tensor3d = layer3d_s[i]->forward(tensor3d);
		}

		tensor_1d tensor1d = layer3to1d->forward(tensor3d);

		for(int i = 0; i < layer1d_len; ++i)
		{
			tensor1d = layer1d_s[i]->forward(tensor1d);
		}

		outputs = tensor1d;
		return outputs;
	}

	tensor_3d backward(tensor_1d outputs_true)
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

		tensor_3d tensor3d = layer3to1d->backward(tensor1d);

		for(int i = layer3d_len - 1; i >= 0; --i)
		{
			tensor3d = layer3d_s[i]->backward(tensor3d);
		}

		return tensor3d;
	}

	void fit(int t, AdamOptimizer& adam)
	{
		for(int i = 0; i < layer3d_len; ++i)
		{
			layer3d_s[i]->fit(t, adam);
		}

		for(int i = 0; i < layer1d_len; ++i)
		{
			layer1d_s[i]->fit(t, adam);
		}
	}
};

#endif