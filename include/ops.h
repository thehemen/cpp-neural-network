#include <omp.h>  // OpenMP headers
#include <tensors/tensor_1d.h>
#include <tensors/tensor_2d.h>

#ifndef OPS_H
#define OPS_H

tensor_2d zeropadding2d(tensor_2d inputs, int width_pad, int height_pad)
{
	int width = inputs.length();
	int height = inputs.width();

	int out_width = width + width_pad * 2;
	int out_height = height + height_pad * 2;

	tensor_2d outputs(out_width, out_height);

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
	int width = inputs.length();
	int height = inputs.width();
	tensor_2d outputs(width, height);

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

tensor_2d conv2d(tensor_2d image, tensor_2d kernel)
{
	int image_width = image.length();
	int image_height = image.width();

	int kernel_width = kernel.length();
	int kernel_height = kernel.width();

	int out_width = image_width - kernel_width + 1;
	int out_height = image_height - kernel_height + 1;

    tensor_2d out(out_width, out_height);

    #pragma omp parallel for
    for(int x = 0; x < out_width; ++x)
    {
        for(int y = 0; y < out_height; ++y)
        {
            for(int k_x = 0; k_x < kernel_width; ++k_x)
            {
                for(int k_y = 0; k_y < kernel_height; ++k_y)
                {
                    out[x][y] += image[x + k_x][y + k_y] * kernel[k_x][k_y];
                }
            }
        }
    }

    return out;
}

#endif