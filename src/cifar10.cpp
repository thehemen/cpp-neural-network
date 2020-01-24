#include <iostream>
#include <ctime>
#include <map>
#include <activation.h>
#include <adam_optimizer.h>
#include <metrics.h>
#include <status.h>
#include <network/network_3to1d.h>
#include <network/network_builder.h>
#include <dataset/cifar10.h>

using namespace std;

int main(int argc, char *argv[])
{
	if(argc != 7)
	{
		cout << "./cifar10 [data-batch-1] ... [data-batch-5] [test-batch]" << endl;
		return 1;
	}

	srand(time(0));

	const double learning_rate = 0.0001;
	const double beta_1 = 0.9;
	const double beta_2 = 0.999;
	const double epsilon = 1e-9;

	const int epochs = 10;

	const int iteration_step = 100;
	const int precision = 6;
	const int space_count = 100;

	Activation relu(ActivationType::RELU);
	AdamOptimizer adam(learning_rate, beta_1, beta_2, epsilon);

	// Divide train dataset by 80%/20%.
	vector<Sample3to1D> train_samples;
	for(int i = 0; i < 4; ++i)
	{
		vector<Sample3to1D> train_batch = get_cifar10_samples(argv[i + 1]);
		train_samples.insert(train_samples.end(), train_batch.begin(), train_batch.end());
	}
	int train_num = train_samples.size();

	vector<Sample3to1D> val_samples = get_cifar10_samples(argv[5]);
	int val_num = val_samples.size();

	vector<Sample3to1D> test_samples = get_cifar10_samples(argv[6]);
	int test_num = test_samples.size();

	// LeNet-like network
	NetworkBuilder networkBuilder(32, 32, 3);
	networkBuilder.add("Conv2D", map<string, int>{{"count", 4}, {"width", 3}, {"height", 3}});
	networkBuilder.add("Activation3D", relu);
	networkBuilder.add("MaxPooling2D", map<string, int>{{"width", 2}, {"height", 2}});
	networkBuilder.add("Conv2D", map<string, int>{{"count", 8}, {"width", 3}, {"height", 3}});
	networkBuilder.add("Activation3D", relu);
	networkBuilder.add("MaxPooling2D", map<string, int>{{"width", 2}, {"height", 2}});
	networkBuilder.add("Flatten");
	networkBuilder.add("Dense", map<string, int>{{"length", 64}});
	networkBuilder.add("Activation1D", relu);
	networkBuilder.add("Dense", map<string, int>{{"length", 10}});
	networkBuilder.add("Softmax");
	Network3to1D network(networkBuilder.get_3d(), networkBuilder.get_3to1d(), networkBuilder.get_1d());
	cout << networkBuilder.get_shapes() << endl;

	Status status(iteration_step, precision, space_count);
	status.initialize();

	for(int i = 0; i < epochs; ++i)
	{
		status.reset(train_num);
		for(int j = 0; j < train_num; ++j)
		{
			int t = i * train_num + j;
			network.forward(train_samples[j].inputs);
			network.backward(train_samples[j].outputs);
			network.fit(t, adam);
			status.update(i, j);
		}

		double loss = 0.0;
		double acc = 0.0;

		status.reset(val_num);
		for(int j = 0; j < val_num; ++j)
		{
			auto results = network.forward(val_samples[j].inputs);
			loss += categorical_crossentropy(val_samples[j].outputs, results);
			acc += categorical_accuracy(val_samples[j].outputs, results);
			status.update(i, j);
		}

		loss /= val_num;
		acc /= val_num;
		status.summarize(i, loss, acc);
	}

	cout << endl;
	double test_loss = 0.0;
	double test_acc = 0.0;

	status.reset(test_num);
	for(int j = 0; j < test_num; ++j)
	{
		auto results = network.forward(test_samples[j].inputs);
		test_loss += categorical_crossentropy(test_samples[j].outputs, results);
		test_acc += categorical_accuracy(test_samples[j].outputs, results);
		status.update(epochs - 1, j);
	}

	test_loss /= test_num;
	test_acc /= test_num;
	status.summarize(epochs - 1, test_loss, test_acc);
	return 0;
}