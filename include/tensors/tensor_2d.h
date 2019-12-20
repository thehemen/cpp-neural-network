#include <tensors/tensor_1d.h>

#ifndef TENSOR_2D_H
#define TENSOR_2D_H

using namespace std;

class tensor_2d
{
	vector<tensor_1d> vec;
public:
	tensor_2d()
	{
		vec = vector<tensor_1d>();
	}

	tensor_2d(int n, int m)
	{
		vec = vector<tensor_1d>(n, tensor_1d(m));
	}

	tensor_2d(vector<vector<double>> vec)
	{
		this->vec = vector<tensor_1d>();

		for(int i = 0, n = vec.size(); i < n; ++i)
		{
			this->vec.push_back(tensor_1d(vec[i]));
		}
	}

	int size()
	{
		return vec.size();
	}

	void make_zero()
	{
		for(int i = 0, n = vec.size(); i < n; ++i)
		{
			vec[i].make_zero();
		}
	}

	void make_random()
	{
		for(int i = 0, n = vec.size(); i < n; ++i)
		{
			vec[i].make_random();
		}
	}

	tensor_1d& operator[](int index)
	{
		return vec[index];
	}
};

#endif