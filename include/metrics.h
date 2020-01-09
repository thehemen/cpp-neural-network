#include <cmath>
#include <cfloat>
#include <types.h>

#ifndef METRIC_H
#define METRIC_H

const double EPSILON = 1e-9;

double binary_crossentropy(tensor_1d y, tensor_1d p)
{
	return -(y[0] * log(p[0] + EPSILON) + (1 - y[0]) * log(1 - p[0] + EPSILON));
}

double categorical_crossentropy(tensor_1d y, tensor_1d p)
{
	double loss = 0.0;

	for(int i = 0, len = y.size(); i < len; ++i)
	{
		loss += y[i] * log(p[i] + EPSILON);
	}

	return -loss;
}

double binary_accuracy(tensor_1d y, tensor_1d p)
{
	return y[0] == round(p[0]) ? 1.0 : 0.0;
}

double categorical_accuracy(tensor_1d y, tensor_1d p)
{
	double y_max = -DBL_MAX;
	int y_argmax = 0;

	double p_max = -DBL_MAX;
	int p_argmax = 1;

	for(int i = 0, len = y.size(); i < len; ++i)
	{
		if(y[i] > y_max)
		{
			y_max = y[i];
			y_argmax = i;
		}

		if(p[i] > p_max)
		{
			p_max = p[i];
			p_argmax = i;
		}
	}

	return y_argmax == p_argmax ? 1.0 : 0.0;
}

#endif