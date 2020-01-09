#include <map>
#include <omp.h>  // OpenMP headers
#include <types.h>
#include <layers/three_to_one_dim/layer3to1d.h>

#ifndef FLATTEN_H
#define FLATTEN_H

using namespace std;

class Flatten : public Layer3to1D
{
	int count;
	int width;
	int height;
public:
	Flatten() : Layer3to1D() {}

	Flatten(map<string, int> params)
	{
		count = params["count"];
		width = params["width"];
		height = params["height"];
	}

	tensor_1d forward(tensor_3d feature_map)
	{
		tensor_1d flatten(count * width * height);

		#pragma omp parallel for
		for(int i = 0; i < count; ++i)
		{
			for(int x = 0; x < width; ++x)
			{
				for(int y = 0; y < height; ++y)
				{
					flatten[i * width * height + x * width + y] = feature_map[i][x][y];
				}
			}
		}

		return flatten;
	}

	tensor_3d backward(tensor_1d gradients)
	{
		tensor_3d feature_map(count, tensor_2d(width, tensor_1d(height)));

		#pragma omp parallel for
		for(int i = 0; i < count; ++i)
		{
			for(int x = 0; x < width; ++x)
			{
				for(int y = 0; y < height; ++y)
				{
					feature_map[i][x][y] = gradients[i * width * height + x * width + y];
				}
			}
		}

		return feature_map;
	}
};

#endif