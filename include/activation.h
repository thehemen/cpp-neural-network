#include <cmath>

#ifndef ACTIVATION_H
#define ACTIVATION_H

enum ActivationType {SIGMOID, TANH, RELU};

class Activation
{
	ActivationType activationType;
	double alpha;
public:
	Activation() {}

	Activation(ActivationType activationType, double alpha = 0.0)
	{
		this->activationType = activationType;
		this->alpha = alpha;
	}

	double get(double arg)
	{
		switch(activationType)
		{
			case ActivationType::SIGMOID:
				return 1.0 / (1.0 + exp(-alpha * arg));

			case ActivationType::TANH:
				return (exp(2.0 * arg) - 1) / (exp(2.0 * arg) + 1);

			case ActivationType::RELU:
				return max(0.0, arg);

			default:
				return 0.0;
		}
	}

	double der(double res, double arg)
	{
		switch(activationType)
		{
			case ActivationType::SIGMOID:
				return alpha * res * (1.0 - res);

			case ActivationType::TANH:
				return 1.0 - res * res;

			case ActivationType::RELU:
				return arg >= 0.0 ? 1.0 : 0.0;

			default:
				return 0.0;
		}
	}
};

#endif