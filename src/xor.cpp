#include <iostream>
#include <ctime>
#include <map>
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

	AdamOptimizer adam(0.1, 0.9, 0.999, 1e-9);
	vector<Sample1D> samples = get_xor_samples();
	int sample_num = samples.size();

	vector<LayerDescription> layers;
	layers.push_back(LayerDescription("dense", map<string, int>{{"length", 8}}, tanh));
	layers.push_back(LayerDescription("dense", map<string, int>{{"length", 1}}, sigm));
	Network network(layers, map<string, int>{{"count", 2}});

	cout << network.get_shapes() << endl;
	cout.precision(6);
	cout << "t:\tLoss:\t\tAccuracy:" << endl;

	int epochs = 10000;
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

		if((i + 1) % 500 == 0)
		{
			for(int j = 0; j < sample_num; ++j)
			{
				tensor_1d results = network.forward_1to1d(samples[j].inputs);
				loss += binary_crossentropy(samples[j].outputs, results);
				acc += binary_accuracy(samples[j].outputs, results);
			}

			loss /= sample_num;
			acc /= sample_num;
			cout << (i + 1) << "\t" << fixed << loss << "\t" << fixed << acc << endl;
		}
	}
	return 0;
}