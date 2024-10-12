#include "error.cuh"
#include "floats.hpp"
#include <chrono>
using namespace std::chrono;
__constant__ int BLOCK_DIM = 128;
real reduce_cpu(const real *x, const int N){
    real sum = 0.0;
    for (int i = 0; i < N; ++i){
        sum += x[i];
    }
    return sum;
}
__global__ void reduce(real *x, real *y){
    const int tid = threadIdx.x;
    real *curr_x = x + blockIdx.x * blockDim.x;
    for (int offset = blockDim.x >> 1; offset > 0; offset >>=1){
        if(tid < offset){
            curr_x[tid] += curr_x[tid + offset];
        }
        __syncthreads();
    }
    if(tid == 0){
        y[blockIdx.x] = curr_x[0];
    }
}

__global__ void reduce_shared(const real *x, real *y, const int N){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int ind = bid * blockDim.x + tid;
    __shared__ real s_x[128];
    s_x[tid] = (ind < N) ? x[ind] : 0.0;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1){
        if(ind < offset){
            s_x[tid] += s_x[tid + offset];
        }
        __syncthreads();
    }
    if(tid == 0){
        y[bid] = s_x[0];   // 保存各个线程块中共享内存的0元素到全局内存
    }
}

__global__ void reduce_shared2(const real *x, real *y, const int N){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int ind = bid * blockDim.x + tid;
    extern __shared__ real s_x[];
    s_x[tid] = (ind < N) ? x[ind] : 0.0;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>=1){
        if(ind << offset){
            s_x[tid] += s_x[tid + offset];
        }
        __syncthreads();
    }
    if(tid == 0){
        y[bid] = s_x[0];
    }
}

int main(){
    unsigned int  N = 1e8;  // 单精度会发生 "大数吃小数" 的现象，导致结果完全错误，双精度没问题
    int M = N * sizeof(real);
    int block_size = 0;
    // CHECK(cudaMemcpyFromSymbol(&block_size, BLOCK_DIM, sizeof(real)));
    CHECK(cudaMemcpyFromSymbol(&block_size, BLOCK_DIM, sizeof(int)));
    int grid_size = (N + block_size - 1) / block_size;
    real *h_x = new real[N];
    real *h_y = new real[grid_size];
    for (int i = 0; i < N; ++i){
        h_x[i] = 1.23;
    }
    cout << "FLOAT_PREC = " << FLOAT_PREC << endl;
    auto t1 = system_clock::now();
    cout << "cpu reduce: " << reduce_cpu(h_x, N) << endl;
    auto t2 = system_clock::now();
    double time = duration<double, std::milli>(t2 - t1).count();
    cout << "cpu reduce time cost: " << time << " ms" << endl;

    real *d_x, *d_y;
    int size = grid_size * sizeof(real);
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMalloc(&d_y, size));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyDefault));
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    cudaEventQuery(start);
    reduce<<<grid_size, block_size>>>(d_x, d_y);
    CHECK(cudaMemcpy(h_y, d_y, size, cudaMemcpyDefault));
    CHECK(cudaGetLastError());
    float elap_time = 0, curr_time = 0;
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));
    cout << "gpu reduce: " << reduce_cpu(h_y, grid_size) << endl;
    printf("gpu reduce time cost: %f ms\n", curr_time - elap_time);
    elap_time = curr_time;

    reduce_shared<<<grid_size, block_size>>>(d_x, d_y, N);
    CHECK(cudaMemcpy(h_y, d_y, size, cudaMemcpyDefault));
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));
    cout << "gpu shared reduce: " << reduce_cpu(h_y, grid_size) << endl;
    printf("gpu shared reduce time cost: %f ms\n", curr_time - elap_time);
    elap_time = curr_time;

    int sharedMemSize = block_size * sizeof(real);  // 核函数中每个线程块的动态共享内存大小
    reduce_shared2<<<grid_size, block_size, sharedMemSize>>>(d_x, d_y, N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_y, d_y, size, cudaMemcpyDefault));
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));
    cout << "gpu shared2 reduce: " << reduce_cpu(h_y, grid_size) << endl;
    printf("gpu shared2 reduce time cost: %f ms\n", curr_time - elap_time);
    elap_time = curr_time;

    delete[] h_x;
    delete[] h_y;
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    return 0;
}
