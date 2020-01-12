#include <iostream>
#include <ctime>
#include <map>
#include <activation.h>
#include <adam_optimizer.h>
#include <metrics.h>
#include <status.h>
#include <network/network_1to2to1d.h>
#include <network/network_builder.h>
#include <dataset/sst.h>

using namespace std;

int main(int argc, char *argv[])
{
	if(argc != 5)
	{
		cout << "./sst [is_binary] [train] [dev] [test]" << endl;
		return 1;
	}

	srand(time(0));

	const double learning_rate = 0.001;
	const double beta_1 = 0.9;
	const double beta_2 = 0.999;
	const double epsilon = 1e-9;

	const int epochs = 10;

	const int iteration_step = 100;
	const int precision = 6;
	const int space_count = 100;

	bool is_binary = stoi(argv[1]) == 1 ? true : false;
	int max_words = 20000;
	int max_len = 32;
	int embedding_dim = 16;

	Activation relu(ActivationType::RELU);
	Activation sigmoid(ActivationType::SIGMOID, 1.0);
	AdamOptimizer adam(learning_rate, beta_1, beta_2, epsilon);

	vector<vector<string>> train_tokens = get_raw_tokens(argv[2]);
	vector<vector<string>> dev_tokens = get_raw_tokens(argv[3]);
	vector<vector<string>> test_tokens = get_raw_tokens(argv[4]);

	vector<vector<string>> all_tokens;
	all_tokens.insert(all_tokens.end(), train_tokens.begin(), train_tokens.end());
	all_tokens.insert(all_tokens.end(), dev_tokens.begin(), dev_tokens.end());

	map<string, int> token_dict = get_token_dict(all_tokens, max_words);
	vector<Sample1D> train_samples = get_sst_samples(train_tokens, token_dict, max_len, is_binary);
	vector<Sample1D> dev_samples = get_sst_samples(dev_tokens, token_dict, max_len, is_binary);
	vector<Sample1D> test_samples = get_sst_samples(test_tokens, token_dict, max_len, is_binary);

	int train_num = train_samples.size();
	int dev_num = dev_samples.size();
	int test_num = test_samples.size();

	NetworkBuilder networkBuilder(max_len);
	networkBuilder.add("Embedding", map<string, int>{{"width", embedding_dim}, {"max_words", max_words}});
	networkBuilder.add("SpatialDropout1D", map<string, float>{{"share", 0.25}});
	networkBuilder.add("Conv1D", map<string, int>{{"count", 8}, {"width", 3}});
	networkBuilder.add("Activation2D", relu);
	networkBuilder.add("GlobalMaxPooling1D");

	if(is_binary)
	{
		networkBuilder.add("Dense", map<string, int>{{"length", 1}});
		networkBuilder.add("Activation1D", sigmoid);
	}
	else
	{
		networkBuilder.add("Dense", map<string, int>{{"length", 5}});
		networkBuilder.add("Softmax");
	}

	Network1to2to1D network(networkBuilder.get_1to2d(), 
		networkBuilder.get_2d(),
		networkBuilder.get_2to1d(), 
		networkBuilder.get_1d());
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

		double dev_loss = 0.0;
		double dev_acc = 0.0;

		status.reset(dev_num);
		for(int j = 0; j < dev_num; ++j)
		{
			auto results = network.forward(dev_samples[j].inputs);

			if(is_binary)
			{
				dev_loss += binary_crossentropy(dev_samples[j].outputs, results);
				dev_acc += binary_accuracy(dev_samples[j].outputs, results);
			}
			else
			{
				dev_loss += categorical_crossentropy(dev_samples[j].outputs, results);
				dev_acc += categorical_accuracy(dev_samples[j].outputs, results);
			}

			status.update(i, j);
		}

		dev_loss /= dev_num;
		dev_acc /= dev_num;
		status.summarize(i, dev_loss, dev_acc);
	}

	cout << endl;
	double test_loss = 0.0;
	double test_acc = 0.0;

	status.reset(test_num);
	for(int i = 0; i < test_num; ++i)
	{
		auto results = network.forward(test_samples[i].inputs);

		if(is_binary)
		{
			test_loss += binary_crossentropy(test_samples[i].outputs, results);
			test_acc += binary_accuracy(test_samples[i].outputs, results);
		}
		else
		{
			test_loss += categorical_crossentropy(test_samples[i].outputs, results);
			test_acc += categorical_accuracy(test_samples[i].outputs, results);
		}

		status.update(epochs - 1, i);
	}

	test_loss /= test_num;
	test_acc /= test_num;
	status.summarize(epochs - 1, test_loss, test_acc);
	return 0;
}