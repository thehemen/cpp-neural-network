#include <vector>
#include <fstream>
#include <types.h>

#ifndef CIFAR10_H
#define CIFAR10_H

using namespace std;

vector<Sample3to1D> get_cifar10_samples(string path)
{
    const int num = 10000;
    const int channels = 3;
	const int width = 32;
	const int height = 32;
    const int classes = 10;

	vector<Sample3to1D> samples;

    ifstream ifile(path, ios::binary);

    if(ifile)
    {
        for(int i = 0; i < num; ++i)
        {
            tensor_3d inputs(channels, tensor_2d(width, tensor_1d(height)));
    		tensor_1d outputs(classes);

            int class_id = ifile.get();
            outputs[class_id] = 1.0;
            
            for(int c = 0; c < channels; ++c)
            {
                for(int x = 0; x < width; ++x)
                {
                    for(int y = 0; y < height; ++y)
                    {
                        int pixel = ifile.get();
                        inputs[c][x][y] = pixel;
                    }
                }
            }

            samples.push_back(Sample3to1D(inputs, outputs));
        }

        ifile.close();
    }

	return samples;
}

#endif