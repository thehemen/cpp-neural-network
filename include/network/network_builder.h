#include <vector>
#include <sstream>
#include <cmath>
#include <map>

#include <activation.h>
#include <types.h>

#include <layers/one_dimensional/layer1d.h>
#include <layers/one_dimensional/dense.h>
#include <layers/one_dimensional/activation1d.h>
#include <layers/one_dimensional/softmax.h>

#include <layers/one_to_two_dim/layer1to2d.h>
#include <layers/one_to_two_dim/embedding.h>

#include <layers/two_dimensional/layer2d.h>
#include <layers/two_dimensional/conv1d.h>
#include <layers/two_dimensional/maxpooling1d.h>
#include <layers/two_dimensional/activation2d.h>

#include <layers/two_to_one_dim/layer2to1d.h>
#include <layers/two_to_one_dim/globalmaxpooling1d.h>

#include <layers/three_to_one_dim/layer3to1d.h>
#include <layers/three_to_one_dim/flatten.h>

#include <layers/three_dimensional/layer3d.h>
#include <layers/three_dimensional/conv2d.h>
#include <layers/three_dimensional/separableconv2d.h>
#include <layers/three_dimensional/maxpooling2d.h>
#include <layers/three_dimensional/activation3d.h>

#ifndef NETWORK_BUILDER_H
#define NETWORK_BUILDER_H

using namespace std;

class NetworkBuilder
{
	int input_count;
	int input_width;
	int input_height;

	vector<Layer3D*> layer3d_s;
	vector<Layer2D*> layer2d_s;
	vector<Layer1D*> layer1d_s;

	Layer3to1D* layer3to1d;
	Layer1to2D* layer1to2d;
	Layer2to1D* layer2to1d;
	
	stringstream ostream;

	int total_params;
public:
	NetworkBuilder(int count)
	{
		input_count = count;
		input_width = 1;
		input_height = 1;
		total_params = 0;

		layer3d_s = vector<Layer3D*>();
		layer2d_s = vector<Layer2D*>();
		layer1d_s = vector<Layer1D*>();

		ostream << "Layer Name:\tParameters:\tParam Count:\tOutput Shape:\n";
		ostream << "\t\t\t\t\t\t" << input_count << "\n";
	}

	NetworkBuilder(int width, int height)
	{
		input_count = 1;
		input_width = width;
		input_height = height;
		total_params = 0;

		layer3d_s = vector<Layer3D*>();
		layer2d_s = vector<Layer2D*>();
		layer1d_s = vector<Layer1D*>();

		ostream << "Layer Name:\tParameters:\tParam Count:\tOutput Shape:\n";
		ostream << "\t\t\t\t\t\t" << input_count << "x" << input_width << "x" << input_height << "\n";
	}

	string get_shapes()
	{
		ostream << "\nTotal params: " << total_params << "\n";
		return ostream.str();
	}

	vector<Layer3D*> get_3d()
	{
		return layer3d_s;
	}

	vector<Layer2D*> get_2d()
	{
		return layer2d_s;
	}

	vector<Layer1D*> get_1d()
	{
		return layer1d_s;
	}

	Layer3to1D* get_3to1d()
	{
		return layer3to1d;
	}

	Layer1to2D* get_1to2d()
	{
		return layer1to2d;
	}

	Layer2to1D* get_2to1d()
	{
		return layer2to1d;
	}

	void add(string layerType)
	{
		if(layerType == "Flatten")
		{
			flatten();
		}
		else if(layerType == "GlobalMaxPooling1D")
		{
			globalmaxpooling1d();
		}
		else if(layerType == "Softmax")
		{
			softmax();
		}
	}

	void add(string layerType, map<string, int> params)
	{
		if(layerType == "Conv2D")
		{
			conv2d(params);
		}
		else if(layerType == "SeparableConv2D")
		{
			separableconv2d(params);
		}
		else if(layerType == "MaxPooling2D")
		{
			maxpooling2d(params);
		}
		else if(layerType == "Embedding")
		{
			embedding(params);
		}
		else if(layerType == "Conv1D")
		{
			conv1d(params);
		}
		else if(layerType == "MaxPooling1D")
		{
			maxpooling1d(params);
		}
		else if(layerType == "Dense")
		{
			dense(params);
		}
	}

	void add(string layerType, Activation activation)
	{
		if(layerType == "Activation3D")
		{
			activation3d(activation);
		}
		else if(layerType == "Activation2D")
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
		make_random(kernel, 1.0 / sqrt(input_count * width * height));

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

		Conv2D* conv2d = new Conv2D(kernel, params);
		layer3d_s.push_back(conv2d);

		int parameter_count = count * input_count * width * height;
		total_params += parameter_count;

		input_count = out_count;
		input_width = out_width;
		input_height = out_height;

		ostream << left << setw(16) << "Conv2D";

		stringstream pstream;
		pstream << count << "x" << width << "x" << height;
		ostream << left << setw(16) << pstream.str();

		ostream << left << setw(16) << parameter_count;
		ostream << input_count << "x" << input_width << "x" << input_height << "\n";
	}

	void separableconv2d(map<string, int> params)
	{
		int kernel_count = params["count"];
		int kernel_width = params["width"];
		int kernel_height = params["height"];

		tensor_4d depthwise_kernel(input_count, tensor_3d(1, tensor_2d(kernel_width, tensor_1d(kernel_height))));
		tensor_4d pointwise_kernel(kernel_count, tensor_3d(input_count, tensor_2d(1, tensor_1d(1))));

		make_random(depthwise_kernel, 1.0 / sqrt(kernel_width * kernel_height));
		make_random(pointwise_kernel, 1.0 / sqrt(input_count));

		params["input_count"] = input_count;
		params["input_width"] = input_width;
		params["input_height"] = input_height;

		int out_count = kernel_count;
		int out_width = input_width - kernel_width + 1;
		int out_height = input_height - kernel_height + 1;

		params["out_width"] = out_width;
		params["out_height"] = out_height;

		params["padding_width"] = input_width - out_width;
		params["padding_height"] = input_height - out_height;

		SeparableConv2D* separableconv2d = new SeparableConv2D(depthwise_kernel, pointwise_kernel, params);
		layer3d_s.push_back(separableconv2d);

		int parameter_count = input_count * kernel_width * kernel_height + kernel_count * input_count;
		total_params += parameter_count;

		input_count = out_count;
		input_width = out_width;
		input_height = out_height;

		ostream << left << setw(16) << "SeparableConv2D";

		stringstream pstream;
		pstream << kernel_count << "x" << kernel_width << "x" << kernel_height;
		ostream << left << setw(16) << pstream.str();

		ostream << left << setw(16) << parameter_count;
		ostream << input_count << "x" << input_width << "x" << input_height << "\n";
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
		layer3d_s.push_back(maxpool2d);

		input_width = out_width;
		input_height = out_height;

		ostream << left << setw(16) << "MaxPooling2D";

		stringstream pstream;
		pstream << width << "x" << height;
		ostream << left << setw(16) << pstream.str();

		ostream << left << setw(16) << "0";
		ostream << input_count << "x" << input_width << "x" << input_height << "\n";
	}

	void activation3d(Activation activation)
	{
		map<string, int> params;
		params["out_count"] = input_count;
		params["out_width"] = input_width;
		params["out_height"] = input_height;

		Activation3D* activation3d = new Activation3D(activation, params);
		layer3d_s.push_back(activation3d);

		ostream << left << setw(16) << activation.get_name();
		ostream << left << setw(16) << "-";
		ostream << left << setw(16) << "0";
		ostream << input_count << "x" << input_width << "x" << input_height << "\n";
	}

	void flatten()
	{
		map<string, int> params;
		params["count"] = input_count;
		params["width"] = input_width;
		params["height"] = input_height;

		Flatten* flatten = new Flatten(params);
		layer3to1d = flatten;

		input_count *= input_width * input_height;
		input_width = 1;
		input_height = 1;

		ostream << left << setw(16) << "Flatten";
		ostream << left << setw(16) << "-";
		ostream << left << setw(16) << "0";
		ostream << input_count << "\n";
	}

	void embedding(map<string, int> params)
	{
		params["count"] = input_count;
		int width = params["width"];
		int max_words = params["max_words"];

		tensor_2d weights(max_words, tensor_1d(width));
		make_random(weights, 1.0 / sqrt(max_words * width));

		Embedding* embedding = new Embedding(weights, params);
		layer1to2d = embedding;

		int parameter_count = width * max_words;
		total_params += parameter_count;

		input_width = params["width"];

		ostream << left << setw(16) << "Embedding";
		ostream << left << setw(16) << "-";
		ostream << left << setw(16) << parameter_count;
		ostream << input_count << "x" << input_width << "\n";
	}

	void conv1d(map<string, int> params)
	{
		int count = params["count"];
		int width = params["width"];

		tensor_3d kernel(count, tensor_2d(input_count, tensor_1d(width)));
		make_random(kernel, 1.0 / sqrt(input_count * width));

		params["input_count"] = input_count;
		params["input_width"] = input_width;

		int out_count = count;
		int out_width = input_width - width + 1;

		params["out_width"] = out_width;
		params["padding_width"] = input_width - out_width;

		Conv1D* conv1d = new Conv1D(kernel, params);
		layer2d_s.push_back(conv1d);

		int parameter_count = count * input_count * width;
		total_params += parameter_count;

		input_count = out_count;
		input_width = out_width;

		ostream << left << setw(16) << "Conv1D";

		stringstream pstream;
		pstream << count << "x" << width;
		ostream << left << setw(16) << pstream.str();

		ostream << left << setw(16) << parameter_count;
		ostream << input_count << "x" << input_width << "\n";
	}

	void maxpooling1d(map<string, int> params)
	{
		int width = params["width"];
		int out_width = input_width / width;

		params["input_count"] = input_count;
		params["input_width"] = input_width;
		params["out_width"] = out_width;

		MaxPooling1D* maxpool1d = new MaxPooling1D(params);
		layer2d_s.push_back(maxpool1d);

		input_width = out_width;

		ostream << left << setw(16) << "MaxPooling1D";

		stringstream pstream;
		pstream << width;
		ostream << left << setw(16) << pstream.str();

		ostream << left << setw(16) << "0";
		ostream << input_count << "x" << input_width << "\n";
	}

	void activation2d(Activation activation)
	{
		map<string, int> params;
		params["count"] = input_count;
		params["width"] = input_width;

		Activation2D* activation2d = new Activation2D(activation, params);
		layer2d_s.push_back(activation2d);

		ostream << left << setw(16) << activation.get_name();
		ostream << left << setw(16) << "-";
		ostream << left << setw(16) << "0";
		ostream << input_count << "x" << input_width << "\n";
	}

	void globalmaxpooling1d()
	{
		map<string, int> params;
		params["count"] = input_count;
		params["width"] = input_width;

		GlobalMaxPooling1D* globalmaxpool1d = new GlobalMaxPooling1D(params);
		layer2to1d = globalmaxpool1d;

		input_width = 1;

		ostream << left << setw(16) << "GlobalMaxPool1D";
		ostream << left << setw(16) << "-";
		ostream << left << setw(16) << "0";
		ostream << input_count << "\n";
	}

	void dense(map<string, int> params)
	{
		int count = params["length"];

		tensor_2d weights(count, tensor_1d(input_count));
		tensor_1d biases(count);

		make_random(weights, 1.0 / sqrt(input_count));
		make_random(biases, 1.0);

		params["input_count"] = input_count;

		Dense* dense = new Dense(weights, biases, params);
		layer1d_s.push_back(dense);

		int parameter_count = count * input_count + count;
		total_params += parameter_count;

		input_count = count;

		ostream << left << setw(16) << "Dense";

		stringstream pstream;
		pstream << count;
		ostream << left << setw(16) << pstream.str();

		ostream << left << setw(16) << parameter_count;
		ostream << input_count << "\n";
	}

	void activation1d(Activation activation)
	{
		map<string, int> params;
		params["out_count"] = input_count;
		Activation1D* activation1d = new Activation1D(activation, params);
		layer1d_s.push_back(activation1d);

		ostream << left << setw(16) << activation.get_name();
		ostream << left << setw(16) << "-";
		ostream << left << setw(16) << "0";
		ostream << input_count << "\n";
	}

	void softmax()
	{
		map<string, int> params;
		params["out_count"] = input_count;
		Softmax* softmax = new Softmax(params);
		layer1d_s.push_back(softmax);

		ostream << left << setw(16) << "Softmax";
		ostream << left << setw(16) << "-";
		ostream << left << setw(16) << "0";
		ostream << input_count << "\n";
	}
};

#endif