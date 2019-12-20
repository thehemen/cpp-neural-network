#include <vector>
#include <algorithm>

#ifndef TENSOR_1D_H
#define TENSOR_1D_H

using namespace std;

double get_random_double()
{
	return rand() / (double)RAND_MAX - 0.5;
}

class tensor_1d
{
	vector<double> vec;
public:
	tensor_1d()
	{
		vec = vector<double>();
	}

	tensor_1d(int n)
	{
		vec = vector<double>(n);
	}

	tensor_1d(vector<double> vec)
	{
		this->vec = vector<double>(vec);
	}

	int size()
	{
		return vec.size();
	}

	void make_zero()
	{
		fill(vec.begin(), vec.end(), 0);
	}

	void make_random()
	{
		generate(vec.begin(), vec.end(), get_random_double);
	}

	double& operator[](int index)
	{
		return vec[index];
	}
};

#endif