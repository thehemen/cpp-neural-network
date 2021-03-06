#include <algorithm>
#include <cstdlib>
#include <omp.h>  // OpenMP headers
#include <types.h>

#ifndef OPS_H
#define OPS_H

double get_random_value(double max_range = 1.0)
{
	return max_range * (double)rand() / RAND_MAX - max_range * 0.5;
}

// Xavier weight initialization.
// Weights are taken randomly within [-1/sqrt(n); 1/sqrt(n)],
// where n is the number of input units.
void make_random(tensor_1d &arr, double max_range)
{
	for(int i = 0, length = arr.size(); i < length; ++i)
	{
		arr[i] = get_random_value(max_range);
	}
}

void make_random(tensor_2d &arr, double max_range)
{
	for(int i = 0, length = arr.size(); i < length; ++i)
	{
		make_random(arr[i], max_range);
	}
}

void make_random(tensor_3d &arr, double max_range)
{
	for(int i = 0, length = arr.size(); i < length; ++i)
	{
		make_random(arr[i], max_range);
	}
}

void make_random(tensor_4d &arr, double max_range)
{
	for(int i = 0, length = arr.size(); i < length; ++i)
	{
		make_random(arr[i], max_range);
	}
}

void make_zero(tensor_1d &arr)
{
	fill(arr.begin(), arr.end(), 0.0);
}

void make_zero(tensor_2d &arr)
{
	for(int i = 0, length = arr.size(); i < length; ++i)
	{
		make_zero(arr[i]);
	}
}

void make_zero(tensor_3d &arr)
{
	for(int i = 0, length = arr.size(); i < length; ++i)
	{
		make_zero(arr[i]);
	}
}

void make_zero(tensor_4d &arr)
{
	for(int i = 0, length = arr.size(); i < length; ++i)
	{
		make_zero(arr[i]);
	}
}

tensor_1d zeropadding1d(tensor_1d inputs, int width_pad)
{
	int width = inputs.size();
	int out_width = width + width_pad * 2;
	tensor_1d outputs(out_width);

	#pragma omp parallel for
	for(int x = width_pad; x < out_width - width_pad; ++x)
	{
		outputs[x] = inputs[x - width_pad];
	}

	return outputs;
}

tensor_1d rot180_1d(tensor_1d inputs)
{
	int width = inputs.size();
	tensor_1d outputs(width);

	#pragma omp parallel for
	for(int x = 0; x < width; ++x)
	{
		outputs[x] = inputs[width - x - 1];
	}

	return outputs;
}

void conv1d(tensor_1d image, tensor_1d kernel, tensor_1d &out)
{
	int image_width = image.size();
	int kernel_width = kernel.size();
	int out_width = image_width - kernel_width + 1;

    #pragma omp parallel for
    for(int x = 0; x < out_width; ++x)
    {
        double acc = 0.0;

        for(int k_x = 0; k_x < kernel_width; ++k_x)
        {
            acc += image[x + k_x] * kernel[k_x];
        }

        out[x] += acc;
    }
}

tensor_2d zeropadding2d(tensor_2d inputs, int width_pad, int height_pad)
{
	int width = inputs.size();
	int height = inputs[0].size();

	int out_width = width + width_pad * 2;
	int out_height = height + height_pad * 2;

	tensor_2d outputs(out_width, tensor_1d(out_height));

	#pragma omp parallel for
	for(int x = width_pad; x < out_width - width_pad; ++x)
	{
		for(int y = height_pad; y < out_height - height_pad; ++y)
		{
			outputs[x][y] = inputs[x - width_pad][y - height_pad];
		}
	}

	return outputs;
}

tensor_2d rot180(tensor_2d inputs)
{
	int width = inputs.size();
	int height = inputs[0].size();
	tensor_2d outputs(width, tensor_1d(height));

	#pragma omp parallel for
	for(int x = 0; x < width; ++x)
	{
		for(int y = 0; y < height; ++y)
		{
			outputs[x][y] = inputs[width - x - 1][height - y - 1];
		}
	}

	return outputs;
}

tensor_2d conv2d(tensor_2d image, tensor_2d kernel, tensor_2d &out)
{
	int image_width = image.size();
	int image_height = image[0].size();

	int kernel_width = kernel.size();
	int kernel_height = kernel[0].size();

	int out_width = image_width - kernel_width + 1;
	int out_height = image_height - kernel_height + 1;

    #pragma omp parallel for
    for(int x = 0; x < out_width; ++x)
    {
        for(int y = 0; y < out_height; ++y)
        {
            double acc = 0.0;

            for(int k_x = 0; k_x < kernel_width; ++k_x)
            {
                for(int k_y = 0; k_y < kernel_height; ++k_y)
                {
                    acc += image[x + k_x][y + k_y] * kernel[k_x][k_y];
                }
            }

            out[x][y] += acc;
        }
    }

    return out;
}

#endif