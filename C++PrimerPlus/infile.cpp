#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cuda_fp16.h>
using namespace std;

int main(){
    // ifstream infile("data.txt");
    // if (infile.is_open()){
    //     std::string line;
    //     while (std::getline(infile, line)){
    //         float f = std::stof(line);
    //         std::cout << f << std::endl;
    //     }
    //     infile.close();
    // } else {
    //     std::cout << "Failed to open file!" << std::endl;
    // }

    // float* x_data = new float[1 * 56 * 56 * 64];
    // std::ifstream infile("data.txt");
    // for (int i = 0; i < 200704; i++){
    //     std::string line;
    //     std::getline(infile, line);
    //     float f = std::stof(line);
    //     std::cout << f << std::endl;
    //     x_data[i] = f;
    // }

    ifstream infile("input", std::ifstream::binary);
    half *data = new half[6291456];
    infile.read((char *)data, 12582912);
    for (int i = 0; i < 6291456; i++){
        std::cout << data[i] << std::endl;
    }

    return 0;
}
