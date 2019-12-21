#include <tensors/tensor_2d.h>

#ifndef TENSOR_3D_H
#define TENSOR_3D_H

using namespace std;

class tensor_3d
{
	vector<tensor_2d> vec;
public:
	tensor_3d()
	{
		vec = vector<tensor_2d>();
	}

	tensor_3d(int n, int m, int t)
	{
		vec = vector<tensor_2d>(n, tensor_2d(m, t));
	}

	tensor_3d(vector<tensor_2d> vec)
	{
		this->vec = vec;
	}

	tensor_3d(vector<vector<vector<double>>> vec)
	{
		this->vec = vector<tensor_2d>();

		for(int i = 0, n = vec.size(); i < n; ++i)
		{
			this->vec.push_back(tensor_2d(vec[i]));
		}
	}

	int size()
	{
		return vec.size();
	}

	int length()
	{
		return vec.size();
	}

	int width()
	{
		if(length() > 0)
		{
			return vec[0].size();
		}
		else
		{
			return 0;
		}
	}

	int height()
	{
		if(width() > 0)
		{
			return vec[0][0].size();
		}
		else
		{
			return 0;
		}
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

	tensor_2d& operator[](int index)
	{
		return vec[index];
	}
};

#endif