#include <vector>
#include <types.h>

#ifndef XOR_H
#define XOR_H

using namespace std;

vector<Sample1D> get_xor_samples()
{
	vector<Sample1D> samples;
	samples.push_back(Sample1D(tensor_1d({0.0, 0.0}), tensor_1d(vector<double>({0.0}))));
	samples.push_back(Sample1D(tensor_1d({0.0, 1.0}), tensor_1d(vector<double>({1.0}))));
	samples.push_back(Sample1D(tensor_1d({1.0, 0.0}), tensor_1d(vector<double>({1.0}))));
	samples.push_back(Sample1D(tensor_1d({1.0, 1.0}), tensor_1d(vector<double>({0.0}))));
	return samples;
}

#endif