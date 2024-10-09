#include <iostream>
using namespace std;
#define N 32
__global__ void reduce(float* a_l){
    int ind = threadIdx.x;
    for (int offset = 16; offset > 1; offset = offset/2){
        if(ind < offset){
            a_l[ind] += a_l[ind + offset];
            __syncthreads();
        }
    }
    if (ind == 0){
        a_l[ind] += a_l[ind + 1];
        printf("===== ind = %d and a_l[%d] = %f \n", ind, ind, a_l[ind]);
    }
}

int main(){
    float *h_a;
    h_a = (float*)malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++){
        h_a[i] = i;
    }
    float sum = 0;
    for (int i = 0; i < N; i++)
    {
        sum += h_a[i];
    }
    cout << "sum = " << sum << endl;
    float *d_a;
    cudaMalloc(&d_a, sizeof(float) * N);
    cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice);
    reduce<<<1, 36>>>(d_a);
    cudaDeviceSynchronize();
    float *res_a;
    res_a = (float *)malloc(sizeof(float) * N);
    cudaMemcpy(res_a, d_a, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cout << "res_a[0] = " << res_a[0] << endl;
    // cout << "res_a[1] = " << res_a[1] << endl;
    free(h_a);
    free(res_a);
    cudaFree(d_a);
    return 0;
}
