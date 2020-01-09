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

	const double learning_rate = 0.1;
	const double beta_1 = 0.9;
	const double beta_2 = 0.999;
	const double epsilon = 1e-9;

	const int epochs = 1000;

	const int iteration_step = 1;
	const int precision = 6;
	const int space_count = 100;

	Activation tanh(ActivationType::TANH);
	Activation sigm(ActivationType::SIGMOID, 1.0);
	AdamOptimizer adam(learning_rate, beta_1, beta_2, epsilon);

	vector<Sample1D> samples = get_xor_samples();
	int sample_num = samples.size();

	NetworkBuilder networkBuilder(2);
	networkBuilder.add("Dense", map<string, int>{{"length", 8}});
	networkBuilder.add("Activation1D", tanh);
	networkBuilder.add("Dense", map<string, int>{{"length", 1}});
	networkBuilder.add("Activation1D", sigm);
	Network network(networkBuilder.get_3d(), networkBuilder.get_3to1d(), networkBuilder.get_1d());
	cout << networkBuilder.get_shapes() << endl;

	Status status(iteration_step, precision, space_count);
	status.initialize();

	for(int i = 0; i < epochs; ++i)
	{
		for(int j = 0; j < sample_num; ++j)
		{
			int t = i * sample_num + j;
			network.forward_1to1d(samples[j].inputs);
			network.backward_1to1d(samples[j].outputs);
			network.fit(t, adam);
		}

		double loss = 0.0;
		double acc = 0.0;

		if(i == 0 || (i + 1) % 50 == 0)
		{
			for(int j = 0; j < sample_num; ++j)
			{
				auto results = network.forward_1to1d(samples[j].inputs);
				loss += binary_crossentropy(samples[j].outputs, results);
				acc += binary_accuracy(samples[j].outputs, results);
			}

			loss /= sample_num;
			acc /= sample_num;
			status.summarize(i, loss, acc);
		}
	}
	return 0;
}