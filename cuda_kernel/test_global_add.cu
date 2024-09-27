#include <iostream>
#include <chrono>
#include <cmath>
#include "error.cuh"
using namespace std::chrono;

__global__ void add(float *x, float *y, float *z, int N){
  // int n = threadIdx.x + blockDim.x * threadIdx.y;
  int n = threadIdx.x + blockDim.x * blockIdx.x;
  /*  // 实现顺序的合并访问
  if (n == 4095){
     printf("=============== threadIdx.x = %d blockIdx.x = %d and blockDim.x = %d \n", threadIdx.x, blockIdx.x, blockDim.x);
  }
  */
  if (n >= N)
    return;
  for (int i = 0; i < 1000; i++){
    z[n] = sqrt(x[n] + y[n]);
  }
}

__global__ void add_permuted(float *x, float *y, float *z, int N){
  int tid = threadIdx.x ^ 0x1;
  /* // 实现乱序的合并访问（相较顺序模式，耗时增加）
  if (blockIdx.x == 4){
    printf("============ tid = %d \n", tid);
  }
  */
  // int n = threadIdx.y * blockDim.x + tid;
  int n = blockIdx.x * blockDim.x + tid;
  if(n>N)return;
  for (int i = 0; i < 1000; ++i){
    z[n] = sqrt(x[n] + y[n]);
  }
}

__global__ void add_offset(float *x, float *y, float *z, int N){
  // 实现不对齐的非合并访问(相较顺序模式,耗时增加)
  // int n = threadIdx.y * blockDim.x + threadIdx.x + 1;
  int n = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if(n >=N) return;
  for (int i = 0; i < 1000; ++i){
    z[n] = sqrt(x[n] + y[n]);
  }
}

__global__ void add_stride(float *x, float *y, float *z, int N){
  // 实现跨越式的非合并访问(相较顺序模式,耗时增加)
  int n = blockIdx.x + threadIdx.x * gridDim.x;
  if(n>=N) return;
  for (int i = 0; i < 1000; ++i){
    z[n] = sqrt(x[n] + y[n]);
  }
}

__global__ void add_broadcast(float *x, float *y, float *z, int N){
  // 实现广播式的非合并访问(相较于顺序模式,耗时增加)
  // int n = threadIdx.x + blockIdx.x * gridDim.x;
  int n = threadIdx.x * threadIdx.y + blockIdx.x;
  if(n>=N) return;
  for (int i = 0; i < 1000; ++i){
    z[n] = sqrt(x[n] + y[n]);
  }
}

void add_cpu(float *x, float *y, float *z, int N){
  int num = 0;
  for (int k = 0; k < N; ++k)
  {
    for (int i = 0; i < 1000; ++i){
      z[k] = sqrt(x[k] + y[k]);
      num++;
    }
  }
  printf("cpu compute num = %d \n", num);
}

__global__ void print_grid_block(){
  int threadx = threadIdx.x;
  int thready = threadIdx.y;
  int threadz = threadIdx.z;
  int blockx = blockIdx.x;
  int blocky = blockIdx.y;
  int blockz = blockIdx.z;
  printf("==== blockx = %d, blocky = %d, blockz = %d, gridDimx = %d, gridDimy = %d, gridDimz = %d\n", blockx, blocky, blockz, gridDim.x, gridDim.y, gridDim.z);
}

int main(){
  int N = 1.0e6;
  int M = N * sizeof(float);
  float *h_x, *h_y, *h_z;
  h_x = new float[N];
  h_y = new float[N];
  h_z = new float[N];
  for (int i = 0; i < N; ++i){
    h_x[i] = 1.0;
    h_y[i] = 2.0;
  }

  // cpu 耗时
  auto t1 = system_clock::now();
  add_cpu(h_x, h_y, h_z, N);
  auto t2 = system_clock::now();
  double time = duration<double, std::milli>(t2 - t1).count();
  std::cout << "cpu time cost = " << time << "ms" << std::endl;

  // device 内存初始化
  float *d_x, *d_y, *d_z;
  CHECK(cudaMalloc(&d_x, M));
  CHECK(cudaMalloc(&d_y, M));
  CHECK(cudaMalloc(&d_z, M));
  CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));
  float elapsed_time = 0;
  float curr_time = 0;
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start));
  cudaEventQuery(start);


  // 顺序合并访问模式（一个线程块有一个线程束，一次请求128字节，如 d_x 中 0-31 个元素）。
  // 若 d_x 的首地址为0，则 0-31 元素的内存分别为 0-3 字节、4-7 字节、... 124-127 字节；
  // 对应 4 次数据传输 0-31 字节、32-63 字节、64-95 字节、96-127 字节，合并度 100%
  add<<<31250, 32>>>(d_x, d_y, d_z, N);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaEventElapsedTime(&curr_time, start, stop));
  printf("add time cost: %f ms.\n", curr_time - elapsed_time);
  elapsed_time = curr_time;
/*
  // 乱序合并访问模式
  add_permuted<<<31250, 32>>>(d_x, d_y, d_z, N);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaEventElapsedTime(&curr_time, start, stop));
  printf("add_permuted time cost : %f ms. \n", curr_time - elapsed_time);
  elapsed_time = curr_time;

  // 不对齐的非合并访问模式（一个线程块依然有一个线程束，一次请求128字节，如 d_x 中 1-32 个元素）.
  // 若 d_x 的首地址为 0, 则 1-32 元素的内存分别为 4-7 字节, ... 124-127, 128-131 字节;
  // 对应 5 次数据传输 0-31 字节, 32-63 字节, 64-95 字节, 96-127 字节, 128-159 字节,
  // 合并度 4*32/(5*32) * 100% = 80%.
  add_offset<<<31250, 32>>>(d_x, d_y, d_z, N);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaEventElapsedTime(&curr_time, start, stop));
  printf("add_offst time cost : %f ms. \n", curr_time - elapsed_time);

  // 跨越式非合并访问模式(一个线程块依然有一个线程束,一次请求 128 字节, 32个元素).
  // 对于第一个线程块,线程束将访问 d_x 中 0, 128, 256, ... 等元素.
  // 因为每个元素都不在一个 32 字节连续内存中,所以将导致 32 次数据传输,
  // 合并度 4*32/(32*32) * 100% = 12.5%
  add_stride<<<31250, 32>>>(d_x, d_y, d_z, N);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaEventElapsedTime(&curr_time, start, stop));
  printf("add_stride time cost : %f ms. \n", curr_time - elapsed_time);
  elapsed_time = curr_time;
*/
  // 广播式非合并访问模式(一个线程块依然有一个线程束,一次请求 128 字节, 32个元素).
  // 对于第一个线程块, 线程束将一致地访问 d_x 中第 0 元素; 所以只产生一次数据传输;
  // 但是线程束只使用了 4个字节,合并度 4/32 * 100% = 12.5%.
  // (这种访问更适合使用常量内存变量.)
  add_broadcast<<<31250, 32>>>(d_x, d_y, d_z, N);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaEventElapsedTime(&curr_time, start, stop));
  printf("add_broadcast time cost : %f ms. \n", curr_time - elapsed_time);
  elapsed_time = curr_time;

  CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyDefault));

  delete[] h_x;
  delete[] h_y;
  delete[] h_z;
  CHECK(cudaFree(d_x));
  CHECK(cudaFree(d_y));
  CHECK(cudaFree(d_z));

  // print_grid_block<<<6, 3>>>();
  // cudaDeviceSynchronize();

  return 0;
}
