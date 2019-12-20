#include <iostream>
#include <ctime>
#include <tensors/tensor_1d.h>
#include <dataset.h>
#include <activation.h>
#include <adam_optimizer.h>
#include <metrics.h>
#include <network.h>

using namespace std;

int main()
{
	srand(time(0));
	Activation tanh(ActivationType::TANH);
	Activation sigm(ActivationType::SIGMOID, 1.0);

	vector<LayerDescription> layers;
	layers.push_back(LayerDescription(tanh, 32));
	layers.push_back(LayerDescription(sigm, 10));
	Network network(768, 10, layers);

	AdamOptimizer adam(0.001, 0.9, 0.999, 1e-9);

	string train_images_path = "../mnist/train-images-idx3-ubyte";
	string train_labels_path = "../mnist/train-labels-idx1-ubyte";
	vector<Sample> train_samples = get_mnist_samples(train_images_path, train_labels_path);
	int train_num = train_samples.size();

	string test_images_path = "../mnist/t10k-images-idx3-ubyte";
	string test_labels_path = "../mnist/t10k-labels-idx1-ubyte";
	vector<Sample> test_samples = get_mnist_samples(test_images_path, test_labels_path);
	int test_num = test_samples.size();

	int epochs = 16;

	for(int i = 0; i < epochs; ++i)
	{
		for(int j = 0; j < train_num; ++j)
		{
			int t = i * train_num + j;
			network.forward(train_samples[j].inputs);
			network.backward(train_samples[j].outputs);
			network.fit(t, adam);
		}

		double loss = 0.0;
		double acc = 0.0;

		for(int j = 0; j < test_num; ++j)
		{
			tensor_1d results = network.forward(test_samples[j].inputs);
			loss += categorical_crossentropy(test_samples[j].outputs, results);
			acc += categorical_accuracy(test_samples[j].outputs, results);
		}

		loss /= test_num;
		acc /= test_num;
		cout << (i + 1) << "\t" << loss << "\t" << acc << endl;
	}
	return 0;
}