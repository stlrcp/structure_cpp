#include <iostream>
#include "error.cuh"
using namespace std;

__global__ void print_ptr(int **h_ptr, int size){
    printf("==============\n");
    for (int i = 0; i < size; i++){
        printf("======= before get h_ptr =====\n");
        printf("========== p_num[%d] = %p\n", i, h_ptr[i]);
    }
    printf("======= after for =======");
}



int main(){
    int **h_a = nullptr;
    cout << " h_a = " << h_a << endl;
    int N = 3;
    cout << "sizeof(int) = " << sizeof(float) << endl;
    cout << "sizeof(int*) = " << sizeof(float*) << endl;
    h_a = (int **)malloc(sizeof(int*) * N);
    cout << " h_a = " << h_a << endl;
    for (int i = 0; i < N; i++){
        int *d_a = nullptr;
        cudaMalloc(&d_a, sizeof(int)*N);
        cout << i << "_d_a = " << d_a << endl;
        h_a[i] = d_a;
    }
    cout << " h_a[0] = " << h_a[0] << endl;

    int **ppd_a = nullptr;
    // cudaMalloc(&ppd_a, sizeof(int)*N);   //  使用 print_ptr 打印的为 （nil）
    cudaMalloc(&ppd_a, sizeof(int*)*N);
    cudaMemcpy(ppd_a, h_a, sizeof(int*)*N, cudaMemcpyHostToDevice);
    cout << "ppd_a = " << ppd_a << endl;
    // cout << "ppd_a = " << *ppd_a << endl;
    // print_ptr<<<1, 1>>>(ppd_a, h_a, N);
    print_ptr<<<1, 1>>>(ppd_a, N);
    cudaDeviceSynchronize();
    cout << "end of print" << endl;
    // getchar();

    float *d1_a, *d2_a;
    cudaMalloc(&d1_a, sizeof(float)*3*N);
    cudaMalloc(&d2_a, sizeof(float)*N);
    CHECK(cudaMemcpy(d2_a, d1_a, sizeof(float)*N, cudaMemcpyDeviceToDevice);)   //  success
    // CHECK(cudaMemcpy(d2_a, d1_a, sizeof(float)*3*N, cudaMemcpyDeviceToDevice);)   //  ERROR invalid argument
    cudaMemcpy(d2_a, d1_a, sizeof(float)*3*N, cudaMemcpyDeviceToDevice);   //  ERROR invalid argument

    int **out_a = nullptr;
    out_a = (int **)malloc(sizeof(int *) * N);
    cudaMemcpy(out_a, ppd_a, sizeof(int*)*N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++){
        cout << " out_a[" << i << "] = " << out_a[i] << endl;
    }

    return 0;
}
