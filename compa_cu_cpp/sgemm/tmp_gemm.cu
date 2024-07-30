#include <iostream>

void __global__ nativegemm(float *m_a, float *m_b, float *m_c, int m, int n, int k){
    int inx = threadIdx.x;
    // for (int i = 0; i < m; i++){
    //     for (int j = 0; j < k; j++){
    //         float psum = 0.0;
    //         if(inx < n){
    //             psum += m_a[i * n + inx] * m_b[inx * k + j];
    //         }
    //         m_c[i * k + j] = psum;
    //     }   // 写法有大问题，if 只能放在 for 外侧
    // }
    if(inx < m){
        for (int j = 0; j < k; j++){
            float psum = 0.0;
            for (int z = 0; z < n; z++)
                psum += m_a[inx * n + z] * m_b[z * k + j];
            m_c[inx * k + j] = psum;
        }
    }
}

void cgemm(int *h_a, int *h_b, int *h_c, int m, int n, int k){
// void cgemm(float *h_a, float *h_b, float *h_c, int m, int n, int k){
    for (int i = 0; i < m; i++){
        for (int j = 0; j < k; j++){
            float psum = 0.0;
            for (int z = 0; z < n; z++)
                psum += h_a[i * n + z] * h_b[z * k + j];
            h_c[i * k + j] = psum;
        }
    }
}

int main(){
    int *h_a, *h_b, *h_c;
    int m = 3;
    int n = 4;
    int k = 2;
    h_a = (int *)malloc(sizeof(int) * 12);
    h_b = (int *)malloc(sizeof(int) * 8);
    h_c = (int *)malloc(sizeof(int) * 6);

    // float *h_a, *h_b, *h_c;
    // int m = 3;
    // int n = 4;
    // int k = 2;
    // h_a = (float *)malloc(sizeof(int) * m*n);
    // h_b = (float *)malloc(sizeof(int) * n*k);
    // h_c = (float *)malloc(sizeof(int) * m*k);

    float *d_a, *d_b, *d_c;
    size_t size_a = sizeof(float) * m * n;
    size_t size_b = sizeof(float) * n * k;
    size_t size_c = sizeof(float) * m * k;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    for (int i = 0; i < 12; i++)
        h_a[i] = i + 1;
    for (int j = 0; j < 8; j++)
        h_b[j] = j + 1;
    // for (int x = 0; x < 6; x++)
    //     h_c[x] = x;
    cgemm(h_a, h_b, h_c, m, n, k);
    for (int z = 0; z < 6; z++){
        printf("%d ", h_c[z]);     //  类型必须保持一致
        if(z % 2 != 0)
            printf("\n");
    }

    for (int x = 0; x < 6; x++)
        h_c[x] = x;

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);    // 虽然 int 和 float 均占4字节，但类型还是需要保持一致
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    nativegemm<<<1, 5>>>(d_a, d_b, d_c, m, n, k);
    
    cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);      //   copy 过程是否存在隐式同步
    cudaDeviceSynchronize();       //   后续研究下是否需要
    for (int z = 0; z < 6; z++)
    {
        printf("%d ", h_c[z]);
        if(z % 2 != 0)
            printf("\n");
    }
}



/*
#include <iostream>
// #include <cuda_runtime.h>

void cgemm(float *h_a, float *h_b, float *h_c, int m, int n, int k){
    for (int i = 0; i < m; i++){
        for (int j = 0; j < k; j++){
            float psum = 0.0;
            for (int z = 0; z < n; z++)
                psum += h_a[i * n + z] * h_b[z * k + j];
            h_c[i * k + j] = psum;
        }
    }
}

void __global__ nativegemm(float *m_a, float *m_b, float *m_c, int m, int n, int k){
    int inx = threadIdx.x;
    // for (int i = 0; i < m; i++){
    //     for (int j = 0; j < k; j++){
    //         float psum = 0.0;
    //         if(inx < n){
    //             psum += m_a[i * n + inx] * m_b[inx * k + j];
    //         }
    //         m_c[i * k + j] = psum;
    //     }   // 写法有大问题，if 只能放在 for 外侧
    // }
    if(inx < m){
        for (int j = 0; j < k; j++){
            float psum = 0.0;
            for (int z = 0; z < n; z++)
                psum += m_a[inx * n + z] * m_b[z * k + j];
            m_c[inx * k + j] = psum;
        }
    }
}

__global__ void nativegemm1(float *m_a, float *m_b, float *m_c, int M, int N, int K){
    int in_a = threadIdx.x;
    int in_b = threadIdx.y;
    if (in_a < M && in_b < K)
    {
        float psum = 0.0;
        for (int z = 0; z < N; z++)
            psum += m_a[in_a * N + z] * m_b[z * K + in_b];
        m_c[in_a * K + in_b] = psum;
    }
}

__global__ void nts_gemm(float *m_a, float *m_b, float *m_c, int M, int N, int K){
    const int m = 3;
    const int n = 4;
    const int k = 2;
    __shared__ float s_a[m][n];
    __shared__ float s_b[n][k];
    int inx = threadIdx.x;
    int iny = threadIdx.y;
    if(inx < M && iny < N){
        for (int z = 0; z < K; z++){
            s_a[inx][iny] = m_a[inx * N + iny];
            s_b[iny][z] = m_b[iny * K + z];
        }
    }
    __syncthreads();
    if(inx<M && iny <K){
        float psum = 0.0;
        for (int z = 0; z < N; z++)
            psum += s_a[inx][z] * s_b[z][iny];
        m_c[inx * K + iny] = psum;
    }
}

int main(){
    float *h_a, *h_b, *h_c;
    int m = 3;
    int n = 4;
    int k = 2;
    h_a = (float *)malloc(sizeof(int) * m*n);
    h_b = (float *)malloc(sizeof(int) * n*k);
    h_c = (float *)malloc(sizeof(int) * m*k);

    float *d_a, *d_b, *d_c;
    size_t size_a = sizeof(float) * m * n;
    size_t size_b = sizeof(float) * n * k;
    size_t size_c = sizeof(float) * m * k;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    for (int i = 0; i < m*n; i++)
        h_a[i] = i + 1;
    for (int j = 0; j < n*k; j++)
        h_b[j] = j + 1;
    
    cgemm(h_a, h_b, h_c, m, n, k);
    for (int z = 0; z < m*k; z++){
        printf("%f ", h_c[z]);
        if(z % k != 0)
            printf("\n");
    }

    for (int x = 0; x < 6; x++)
        h_c[x] = x;

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    // nativegemm<<<1, 5>>>(d_a, d_b, d_c, m, n, k);
    dim3 block3(10, 5);
    // nativegemm1<<<1, block3>>>(d_a, d_b, d_c, m, n, k);
    nts_gemm<<<1, block3>>>(d_a, d_b, d_c, m, n, k);

    cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    for (int z = 0; z < m*k; z++)
    {
        printf("%f ", h_c[z]);
        if(z % k != 0)
            printf("\n");
    }
}
*/


