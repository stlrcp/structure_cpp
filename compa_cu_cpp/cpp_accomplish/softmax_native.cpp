#include <iostream>
#include <cmath>
#include <vector>
using namespace std;

vector<double> csfm(vector<double> veca){
	vector<double> vec_res;
	double res_all;
	for(auto i : veca){
		res_all += exp(i);
	}
	for(auto i : veca){
		double exp_i = exp(i) / res_all;
		vec_res.push_back(exp_i);
	}
	return vec_res;
}

int main(){
	vector<double> t_a = {1.2, 2.3, 3.4, 4.5};
	vector<double> res;
	res = csfm(t_a);
	for(auto i: res){
		cout << " i = " << i << endl;
	}
	cout << "end of out" << endl;
}


/*
// py result
import torch
import torch.nn as nn
inp = torch.Tensor([1.2, 2.3, 3.4, 4.5])
out = nn.Softmax()
output = out(inp)
print(output)
*/
