#include <cmath>

#ifndef ADAM_OPTIMIZER_H
#define ADAM_OPTIMIZER_H

using namespace std;

class AdamOptimizer
{
	double alpha;
	double beta_1;
	double beta_2;
	double epsilon;
public:
	AdamOptimizer(double alpha, double beta_1, double beta_2, double epsilon)
	{
		this->alpha = alpha;
		this->beta_1 = beta_1;
		this->beta_2 = beta_2;
		this->epsilon = epsilon;
	}

	double optimize(int t, double& m_t, double& v_t, double g_t)
	{
		m_t = this->beta_1 * m_t + (1.0 - this->beta_1) * g_t;
		v_t = this->beta_2 * v_t + (1.0 - this->beta_2) * (g_t * g_t);

		double m_t_aver = m_t / (1.0 - pow(this->beta_1, t + 1));
		double v_t_aver = v_t / (1.0 - pow(this->beta_2, t + 1));

		return -(this->alpha * m_t_aver) / (sqrt(v_t_aver) + this->epsilon);
	}
};

#endif