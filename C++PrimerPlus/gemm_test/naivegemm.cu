#include <iostream>
using namespace std;

__global__ void naiveSgemm(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K){
        int m = blockIdx.x * blockDim.x + threadIdx.x;
        int n = blockIdx.y * blockDim.y + threadIdx.y;
        float sum = 0.0f;
        for(int k=0; k<K; k++){
            sum += a[m*K+k] * b[k*N+n];
        }
        c[m*N+n] = sum;
    }

int main(){
     int M=2, N=2, K=3;
     float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;
     size_t size_a = M*K*sizeof(float);
     size_t size_b = K*N*sizeof(float);
     size_t size_c = M*N*sizeof(float);
     h_a = (float*)malloc(size_a);
     h_b = (float*)malloc(size_b);
     h_c = (float*)malloc(size_c);
     for(int i=0; i<6; i++){
          h_a[i] = i;
          h_b[i] = i;
     }
     cudaMalloc(&d_a, size_a);
     cudaMalloc(&d_b, size_b);
     cudaMalloc(&d_c, size_c);

     cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
     cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

     dim3 block(2, 2);
     dim3 grid(1, 1);

     naiveSgemm<<<grid, block>>>(d_a, d_b, d_c, M, N, K);

     cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);

     for(int i=0; i<4; i++)
             cout << h_c[i] << endl;
     free(h_a);
     free(h_b);
     free(h_c);
     cudaFree(d_a);
     cudaFree(d_b);
     cudaFree(d_c);
     return 0;
}
