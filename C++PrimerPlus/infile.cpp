/*
//  reverselist 
#include <iostream>
using namespace std;

struct listNode{
    int num;
    listNode *next;
    listNode(int val) : num(val), next(nullptr){}
};

void print_list(listNode* input){
    while(input != nullptr){
        cout << input->num << endl;
        input = input->next;
    }
}

void turn_list(listNode* input){
    // listNode *head = new listNode(0);
    listNode *head = nullptr;
    listNode *src;
    while(input != nullptr){
        // head = input;
        listNode *tmp = new listNode(input->num);
        src = tmp;
        tmp->next = head;
        head = tmp;
        input = input->next;
    }
    print_list(src);
}

int main(){
    int num[5] = {2, 4, 5, 3, 1};
    listNode *head = new listNode(0);
    listNode *pre;
    pre = head;
    for (auto i : num)
    {
        // cout << "i = " << i << endl;
        head->next = new listNode(i);
        head = head->next;
    }
    // print_list(pre);
    turn_list(pre->next);
    return 0;
}
*/

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
