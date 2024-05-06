#include <iostream>
#include <cstdlib>
#include "nv_kernel.cu"

void checkResult(float *hostRef, const int N) {
    float epsilon = 1.0E-6;
    bool match = 1;
    int num = 0;
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - (float)(0.0)) > epsilon) {
            match = 0;
            // printf("Arrays do not match ! \n");
            // printf("host %5.6f at current %d \n", hostRef[i], i);
            // break;
            num += 1;
        }
    }
    if (match)
        printf("Arrays match. \n\n");
    printf("============ num = %d \n", num);
}

int main() {
    int N = 21*67*21*21;
    int M = 21 * 67;
    float *h_a, *h_c;
    float *h_b, *h_d;
    size_t nBytes = N * sizeof(float);
    size_t mBytes = M * sizeof(float);
    h_a = (float *)malloc(nBytes);
    h_c = (float *)malloc(nBytes);
    h_b = (float *)malloc(mBytes);

    for (int i=0; i<N; i++) {
        h_a[i] = (float)1;
        h_c[i] = (float)0;
    }

    for (int j = 0; j < M; j++){
        h_b[j] = (float)0;
    }

    float *d_a;
    float *d_b;
    float *d_c;
    float *d_d;
    cudaMalloc((float **)&d_a, nBytes);
    cudaMalloc((float **)&d_b, mBytes);
    cudaMalloc((float **)&d_c, nBytes);
    cudaMalloc((float **)&d_d, mBytes);
    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, mBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_b, mBytes, cudaMemcpyHostToDevice);
    Tensor<float, 4> Da;
    Tensor<float, 4> Dc;
    Tensor<float, 2> Db;
    Tensor<float, 2> Dd;
    Da.size[0] = 21;
    Da.size[1] = 67;
    Da.size[2] = 21;
    Da.size[3] = 21;
    Da.data = d_a;
    Dc.size[0] = 21;
    Dc.size[1] = 67;
    Dc.size[2] = 21;
    Dc.size[3] = 21;
    Dc.data = d_c;

    Db.size[0] = 21;
    Db.size[1] = 67;
    Db.data = d_b;
    Dd.size[0] = 21;
    Dd.size[1] = 67;
    Dd.data = d_d;

    int64_t D1 = 21;
    int64_t D2 = 21;

    dim3 grid(88, 1, 1);
    dim3 block(441, 4, 1);
    const int smem = 7056;
    kernel1<<<grid, block, smem>>>(Da, D1, D2, Dc, Db, Dd);
    float *h_d2;
    h_d2 = (float *)malloc(nBytes);
    cudaMemcpy(h_d2, d_c, nBytes, cudaMemcpyDeviceToHost);

    // initTensor<<<1, 1>>>(d_a, &D0);
    // printTensor<<<1, 1>>>(D0);
    // cudaDeviceSynchronize();
    // printTensor<<<1, 1>>>(D1);
    // cudaDeviceSynchronize();
    // printTensor<<<1, 1>>>(D2);
    // std::cout << h_d2[66009] << std::endl;
    checkResult(h_d2, N);
    for (int i = 0; i < N; i++) {
        printf("========= h_d2[i] %d  = %f \n", i, h_d2[i]);
        if (i == 10)
            break;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d2);
    return 0;
}
