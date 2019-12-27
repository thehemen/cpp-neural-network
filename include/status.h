#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>

#ifndef STATUS_H
#define STATUS_H

using namespace std;

class Status
{
	int iteration_step;
	int precision;
	int space_count;
	int max_iterations;

	chrono::time_point<chrono::high_resolution_clock> begin_time;
public:
	Status(int iteration_step, int precision, int space_count)
	{
		this->iteration_step = iteration_step;
		this->precision = precision;
		this->space_count = space_count;
		max_iterations = 1;
	}

	void initialize()
	{
		cout.precision(precision);
		cout << "t:\tLoss:\t\tAccuracy:" << endl;
	}

	void reset(int max_iterations)
	{
		this->max_iterations = max_iterations;
		begin_time = chrono::high_resolution_clock::now();
	}

	void update(int epochs, int iterations)
	{
		if(iterations != 0 && iterations % iteration_step == 0)
		{
			clean();
			double share = (double)iterations / max_iterations;
			auto end_time = chrono::high_resolution_clock::now();
			int ms_elapsed = chrono::duration_cast<chrono::milliseconds>(end_time - begin_time).count();
			int ms_left = (int)((double)ms_elapsed / share * (1.0 - share));
			int ms_full = ms_elapsed + ms_left;
			cout << "\r" << epochs + 1 << " | ";
			cout << (int)(share * 100.0) << "% | ";
			cout << iterations << " / " << max_iterations << " | ";
			cout << get_pretty_time(ms_elapsed, 3) << " elapsed | ";
			cout << get_pretty_time(ms_left, 3) << " left | ";
			cout << get_pretty_time(ms_full, 3) << " full." << flush;
		}
	}

	void summarize(int epochs, double loss, double accuracy)
	{
		clean();
		cout << "\r" << epochs + 1;
		cout << "\t" << fixed << loss;
		cout << "\t" << fixed << accuracy << endl;
	}

private:
	void clean()
	{
		cout << "\r";
		for(int i = 0; i < space_count; ++i)
		{
			cout << " ";
		}
		cout << flush;
	}

	string get_pretty_time(int time_raw, int ms_leading_zeros)
	{
		stringstream out;
		int milliseconds = time_raw % 1000; time_raw /= 1000;
		int seconds = time_raw % 60; time_raw /= 60;
		int minutes = time_raw;
		out << minutes << ":" << setw(2) << setfill('0') << seconds << ".";
		out << setw(ms_leading_zeros) << setfill('0') << milliseconds;
		return out.str();
	}
};

#endif