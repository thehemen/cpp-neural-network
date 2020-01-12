#include <map>
#include <omp.h>
#include <types.h>

#ifndef EMBEDDING_H
#define EMBEDDING_H

using namespace std;

class Embedding : public Layer1to2D
{
	int length;
	int width;
	int max_words;

	tensor_1d inputs;
	tensor_2d gradients;

	tensor_2d weights;
	tensor_2d Mt;
	tensor_2d Vt;
public:
	Embedding(tensor_2d weights, map<string, int> params)
	{
		this->weights = weights;

		length = params["count"];
		width = params["width"];
		max_words = params["max_words"];

		Mt = tensor_2d(max_words, tensor_1d(width));
		Vt = tensor_2d(max_words, tensor_1d(width));
	}

	tensor_2d forward(tensor_1d inputs) override
	{
		this->inputs = inputs;
		tensor_2d outputs(length, tensor_1d(width));

		#pragma omp parallel for
		for(int i = 0; i < length; ++i)
		{
			int token_index = inputs[i] - 1;

			if(token_index >= 0 && token_index < max_words)
			{
				outputs[i] = weights[token_index];
			}
		}

		return outputs;
	}

	tensor_1d backward(tensor_2d gradients) override
	{
		this->gradients = gradients;
		tensor_1d gradient_back(length);

		#pragma omp parallel for
		for(int i = 0; i < length; ++i)
		{
			for(int x = 0; x < width; ++x)
			{
				gradient_back[i] += gradients[i][x];
			}
		}

		return gradient_back;
	}

	void fit(int t, AdamOptimizer& adam) override
	{
		#pragma omp parallel for
		for(int i = 0; i < length; ++i)
		{
			int token_index = inputs[i] - 1;

			if(token_index >= 0 && token_index < max_words)
			{
				for(int x = 0; x < width; ++x)
				{
					double update = adam.optimize(t, Mt[token_index][x], Vt[token_index][x], gradients[i][x]);
					weights[token_index][x] += update;
				}
			}
		}
	}
};

#endif