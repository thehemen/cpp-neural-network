#include <vector>
#include <map>
#include <fstream>
#include <algorithm>
#include <types.h>

#ifndef SST_H
#define SST_H

using namespace std;

vector<vector<string>> get_raw_tokens(string path)
{
	vector<vector<string>> sst_tokens;
	ifstream infile(path);

    if(infile.is_open()) 
    {
    	string line;

		while (getline(infile, line))
		{
		    istringstream istream(line);
		    vector<string> tokens;
		    string token;

		    while(istream >> token)
		    {
		    	tokens.push_back(token);
		    }

		    sst_tokens.push_back(tokens);
		}
    }

    return sst_tokens;
}

map<string, int> get_token_dict(vector<vector<string>> sst_tokens, int max_words)
{
	map<string, int> token_freq;

	for(int i = 0, line_num = sst_tokens.size(); i < line_num; ++i)
	{
		// class id is skipped
		for(int j = 1, token_num = sst_tokens[i].size(); j < token_num; ++j)
		{
			token_freq[sst_tokens[i][j]]++;
		}
	}

	vector<pair<string, int>> token_freq_paired;

	for(const auto & [key, value] : token_freq)
	{
		token_freq_paired.push_back(pair<string, int>(key, value));
	}

	sort(token_freq_paired.begin(), token_freq_paired.end(), [](auto &left, auto &right) {
	    return left.second > right.second;
	});

	map<string, int> token_dict;

	for(int i = 0, len = min(max_words, (int)token_freq_paired.size()); i < len; ++i)
	{
		token_dict[token_freq_paired[i].first] = i + 1;  // zero is used for skipped words
	}

	return token_dict;
}

vector<Sample1D> get_sst_samples(vector<vector<string>> sst_tokens, map<string, int> token_dict, int max_len, bool is_binary)
{
	vector<Sample1D> samples;

	for(int i = 0, line_num = sst_tokens.size(); i < line_num; ++i)
	{
		tensor_1d inputs(max_len);
		
		for(int j = 0, token_num = sst_tokens[i].size() - 1; j < token_num; ++j)
		{
			if(j >= max_len)
			{
				break;
			}

			string word = sst_tokens[i][j + 1];
			int token_id = 0;

			if(token_dict.count(word) != 0)
			{
				token_id = token_dict[word];
			}

			inputs[j] = token_id;
		}

		int class_id = stoi(sst_tokens[i][0]);
		int class_num = is_binary ? 1 : 5;  // five classes for fine-grained
		tensor_1d outputs(class_num);

		if(is_binary)
		{
			//binary crossentropy
			outputs[0] = class_id;
		}
		else
		{
			//categorical crossentropy
			outputs[class_id] = 1.0;
		}

		samples.push_back(Sample1D(inputs, outputs));
	}

	return samples;
}

#endif