#include <iostream>
#include <cmath>
#include <vector>
using namespace std;
#define eps 1e-7

void mean(vector<float> input, float* mean){
	float sum=0; // 必须初始化，不然初始值不固定，影响结果
	for (auto i : input)
	{
		sum += i;
	}
	*mean = sum / input.size();
}
void var(vector<float> input, float* var){
	float m_value;
	mean(input, &m_value);
	float sum_square=0;
	for(auto i : input){
		sum_square += pow((i - m_value), 2);
	}
	*var = sum_square / input.size();
}

vector<float> normalization(vector<float> input, float mean, float var){
	vector<float> output;
	for (auto inp : input)
	{
		float tmp = (inp - mean) / sqrt(var + eps);
		output.push_back(tmp);
	}
	return output;
}

int main(){
	vector<float> inp_a = {1.2, 2.3, 3.4, 4.5};
	float a_mean;
	mean(inp_a, &a_mean);
	// cout << a_mean << endl;
	float a_var;
	var(inp_a, &a_var);
	// cout << a_var << endl;
	vector<float> norm_out;
	norm_out = normalization(inp_a, a_mean, a_var);
	for(auto i : norm_out)
		cout << "out = " << i << endl;
}


/*
import torch
import torch.nn as nn
inp = torch.Tensor([ [1.2], [2.3], [3.4], [4.5]])
mod = nn.BatchNorm1d(1, eps=1e-7, momentum=None, affine=False, track_running_stats=False)
out = mod(inp)
print(mod)
print(out)
*/
