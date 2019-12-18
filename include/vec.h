#include <vector>
#include <algorithm>

#ifndef VEC_H
#define VEC_H

using namespace std;

double double_rand()
{
	return (rand() / (double)RAND_MAX) - 0.5;
}

class vec1d
{
	vector<double> vec;
public:
	vec1d()
	{
		vec = vector<double>();
	}

	vec1d(int n)
	{
		vec = vector<double>(n);
	}

	vec1d(vector<double> vec)
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
		generate(vec.begin(), vec.end(), double_rand);
	}

	double& operator[](int index)
	{
		return vec[index];
	}
};

class vec2d
{
	vector<vec1d> vec;
public:
	vec2d()
	{
		vec = vector<vec1d>();
	}

	vec2d(int n, int m)
	{
		vec = vector<vec1d>(n, vec1d(m));
	}

	vec2d(vector<vector<double>> vec)
	{
		this->vec = vector<vec1d>();

		for(int i = 0, n = vec.size(); i < n; ++i)
		{
			this->vec.push_back(vec1d(vec[i]));
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

	vec1d& operator[](int index)
	{
		return vec[index];
	}
};

class vec3d
{
	vector<vec2d> vec;
public:
	vec3d()
	{
		vec = vector<vec2d>();
	}

	vec3d(int n, int m, int t)
	{
		vec = vector<vec2d>(n, vec2d(m, t));
	}

	vec3d(vector<vector<vector<double>>> vec)
	{
		this->vec = vector<vec2d>();

		for(int i = 0, n = vec.size(); i < n; ++i)
		{
			this->vec.push_back(vec2d(vec[i]));
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

	vec2d& operator[](int index)
	{
		return vec[index];
	}
};

#endif