#include <vector>
#include <fstream>
#include <tensors/tensor_1d.h>
#include <tensors/tensor_2d.h>
#include <tensors/tensor_3d.h>

#ifndef DATASET_H
#define DATASET_H

using namespace std;

struct Sample1D
{
    tensor_1d inputs;
    tensor_1d outputs;

    Sample1D(tensor_1d inputs, tensor_1d outputs)
    {
        this->inputs = inputs;
        this->outputs = outputs;
    }
};

struct Sample2to1D
{
	tensor_3d inputs;
	tensor_1d outputs;

	Sample2to1D(tensor_3d inputs, tensor_1d outputs)
	{
		this->inputs = inputs;
		this->outputs = outputs;
	}
};

u_char** read_images(string full_path, int& number_of_images, int& image_size) 
{
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    ifstream file(full_path, ios::binary);

    if(file.is_open()) 
    {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;
        u_char** _dataset = new u_char*[number_of_images];

        for(int i = 0; i < number_of_images; i++) 
        {
            _dataset[i] = new u_char[image_size];
            file.read((char *)_dataset[i], image_size);
        }

        return _dataset;
    } 
    else 
    {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}

u_char* read_labels(string full_path, int& number_of_labels) 
{
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    ifstream file(full_path, ios::binary);

    if(file.is_open()) 
    {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        u_char* _dataset = new u_char[number_of_labels];

        for(int i = 0; i < number_of_labels; i++) 
        {
            file.read((char*)&_dataset[i], 1);
        }
        return _dataset;

    }
    else
    {
        throw runtime_error("Unable to open file `" + full_path + "`!");
    }
}

vector<Sample2to1D> get_mnist_samples(string images_full_path, string labels_full_path)
{
	const int width = 28;
	const int height = 28;
	const int nums = 10;

	vector<Sample2to1D> samples;

	int number_of_images, number_of_labels, image_size;
    u_char** images = read_images(images_full_path, number_of_images, image_size);
    u_char* labels = read_labels(labels_full_path, number_of_labels);

    for(int i = 0; i < number_of_images; ++i)
    {
        int x = 0, y = 0;
        tensor_3d inputs(1, width, height);
		tensor_1d outputs(10);
        
        for(int j = 0; j < image_size; ++j)
        {
            inputs[0][x][y] = (double)(int)images[i][j] / 256.0;
            y++;

            if(y == 28)
            {
                x++;
                y = 0;
            }
        }

        int class_id = (int)labels[i];
        outputs[class_id] = 1.0;
        samples.push_back(Sample2to1D(inputs, outputs));
    }

	return samples;
}

vector<Sample1D> get_xor_samples()
{
	vector<Sample1D> samples;
	samples.push_back(Sample1D(tensor_1d({0.0, 0.0}), tensor_1d(vector<double>({0.0}))));
	samples.push_back(Sample1D(tensor_1d({0.0, 1.0}), tensor_1d(vector<double>({1.0}))));
	samples.push_back(Sample1D(tensor_1d({1.0, 0.0}), tensor_1d(vector<double>({1.0}))));
	samples.push_back(Sample1D(tensor_1d({1.0, 1.0}), tensor_1d(vector<double>({0.0}))));
	return samples;
}

#endif