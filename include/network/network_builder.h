#include <vector>
#include <sstream>
#include <map>

#include <activation.h>
#include <types.h>

#include <layers/one_dimensional/layer1d.h>
#include <layers/one_dimensional/dense.h>
#include <layers/one_dimensional/activation1d.h>

#include <layers/two_to_one_dim/layer2to1d.h>
#include <layers/two_to_one_dim/flatten.h>

#include <layers/two_dimensional/layer2d.h>
#include <layers/two_dimensional/conv2d.h>
#include <layers/two_dimensional/maxpooling2d.h>
#include <layers/two_dimensional/activation2d.h>

#ifndef NETWORK_BUILDER_H
#define NETWORK_BUILDER_H

using namespace std;

class NetworkBuilder
{
	int input_count;
	int input_width;
	int input_height;

	vector<Layer2D*> layer2d_s;
	Layer2to1D* layer2to1d;
	vector<Layer1D*> layer1d_s;
	stringstream ostream;
public:
	NetworkBuilder(int count)
	{
		input_count = count;
		input_width = 1;
		input_height = 1;

		layer2d_s = vector<Layer2D*>();
		layer1d_s = vector<Layer1D*>();

		ostream << input_count << "\n";
	}

	NetworkBuilder(int width, int height)
	{
		input_count = 1;
		input_width = width;
		input_height = height;

		layer2d_s = vector<Layer2D*>();
		layer1d_s = vector<Layer1D*>();

		ostream << input_width << "x" << input_height << "\n";
	}

	string get_shapes()
	{
		return ostream.str();
	}

	vector<Layer2D*> get_2d()
	{
		return layer2d_s;
	}

	Layer2to1D* get_2to1d()
	{
		return layer2to1d;
	}

	vector<Layer1D*> get_1d()
	{
		return layer1d_s;
	}

	void add(string layerType)
	{
		if(layerType == "Flatten")
		{
			flatten();
		}
	}

	void add(string layerType, map<string, int> params)
	{
		if(layerType == "Conv2D")
		{
			conv2d(params);
		}
		else if(layerType == "MaxPooling2D")
		{
			maxpooling2d(params);
		}
		else if(layerType == "Dense")
		{
			dense(params);
		}
	}

	void add(string layerType, Activation activation)
	{
		if(layerType == "Activation2D")
		{
			activation2d(activation);
		}
		else if(layerType == "Activation1D")
		{
			activation1d(activation);
		}
	}

private:
	void conv2d(map<string, int> params)
	{
		int count = params["count"];
		int width = params["width"];
		int height = params["height"];

		tensor_4d kernel(count, tensor_3d(input_count, tensor_2d(width, tensor_1d(height))));
		tensor_1d biases(count);

		make_random(kernel);
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

		Conv2D* conv2d = new Conv2D(kernel, biases, params);
		layer2d_s.push_back(conv2d);

		input_count = out_count;
		input_width = out_width;
		input_height = out_height;

		ostream << input_count << "x" << input_width << "x" << input_height;
		ostream << "\tConv2D\n";
	}

	void maxpooling2d(map<string, int> params)
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

	void activation2d(Activation activation)
	{
		map<string, int> params;
		params["out_count"] = input_count;
		params["out_width"] = input_width;
		params["out_height"] = input_height;

		Activation2D* activation2d = new Activation2D(activation, params);
		layer2d_s.push_back(activation2d);
	}

	void flatten()
	{
		map<string, int> params;
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

	void dense(map<string, int> params)
	{
		int out_count = params["length"];

		tensor_2d weights(out_count, tensor_1d(input_count));
		tensor_1d biases(out_count);

		make_random(weights);
		make_random(biases);

		params["input_count"] = input_count;

		Dense* dense = new Dense(weights, biases, params);
		layer1d_s.push_back(dense);

		input_count = out_count;

		ostream << input_count;
		ostream << "\tDense\n";
	}

	void activation1d(Activation activation)
	{
		map<string, int> params;
		params["out_count"] = input_count;
		Activation1D* activation1d = new Activation1D(activation, params);
		layer1d_s.push_back(activation1d);
	}
};

#endif