#include <iostream>
#include <ctime>
#include <map>
#include <dataset.h>
#include <activation.h>
#include <adam_optimizer.h>
#include <metrics.h>
#include <status.h>
#include <network/network.h>
#include <network/network_builder.h>

using namespace std;

int main()
{
	srand(time(0));

	const double learning_rate = 0.001;
	const double beta_1 = 0.9;
	const double beta_2 = 0.999;
	const double epsilon = 1e-9;

	const int epochs = 16;

	const int iteration_step = 100;
	const int precision = 6;
	const int space_count = 100;

	Activation relu(ActivationType::RELU);
	Activation sigm(ActivationType::SIGMOID, 1.0);
	AdamOptimizer adam(learning_rate, beta_1, beta_2, epsilon);

	string train_images_path = "../mnist/train-images-idx3-ubyte";
	string train_labels_path = "../mnist/train-labels-idx1-ubyte";
	vector<Sample2to1D> train_samples = get_mnist_samples(train_images_path, train_labels_path);
	int train_num = train_samples.size();

	string test_images_path = "../mnist/t10k-images-idx3-ubyte";
	string test_labels_path = "../mnist/t10k-labels-idx1-ubyte";
	vector<Sample2to1D> test_samples = get_mnist_samples(test_images_path, test_labels_path);
	int test_num = test_samples.size();

	// LeNet5-like network
	NetworkBuilder networkBuilder(28, 28);
	networkBuilder.add("Conv2D", map<string, int>{{"count", 6}, {"width", 3}, {"height", 3}});
	networkBuilder.add("Activation2D", relu);
	networkBuilder.add("MaxPooling2D", map<string, int>{{"width", 2}, {"height", 2}});
	networkBuilder.add("Conv2D", map<string, int>{{"count", 16}, {"width", 3}, {"height", 3}});
	networkBuilder.add("Activation2D", relu);
	networkBuilder.add("MaxPooling2D", map<string, int>{{"width", 2}, {"height", 2}});
	networkBuilder.add("Flatten");
	networkBuilder.add("Dense", map<string, int>{{"length", 10}});
	networkBuilder.add("Activation1D", sigm);
	Network network(networkBuilder.get_2d(), networkBuilder.get_2to1d(), networkBuilder.get_1d());
	cout << networkBuilder.get_shapes() << endl;

	Status status(iteration_step, precision, space_count);
	status.initialize();

	for(int i = 0; i < epochs; ++i)
	{
		status.reset(train_num);
		for(int j = 0; j < train_num; ++j)
		{
			int t = i * train_num + j;
			network.forward_2to1d(train_samples[j].inputs);
			network.backward_1to2d(train_samples[j].outputs);
			network.fit(t, adam);
			status.update(i, j);
		}

		double loss = 0.0;
		double acc = 0.0;

		status.reset(test_num);
		for(int j = 0; j < test_num; ++j)
		{
			auto results = network.forward_2to1d(test_samples[j].inputs);
			loss += categorical_crossentropy(test_samples[j].outputs, results);
			acc += categorical_accuracy(test_samples[j].outputs, results);
			status.update(i, j);
		}

		loss /= test_num;
		acc /= test_num;
		status.summarize(i, loss, acc);
	}
	return 0;
}