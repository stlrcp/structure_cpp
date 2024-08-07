//   ====== 原子函数的合理使用 ======
// cuda中，一个线程的原子操作可以在不受其他线程的任何操作的影响下完成对某个（全局内存或共享内存）
// 数据的一套"读-改-写"操作。
// === 完全在 GPU 中进行规约 ===
// 有两种方法能够在 GPU 中得到最终结果：
//  1. 用另一个核函数将较短的数组进一步规约；
//  2. 在核函数末尾利用原子函数进行规约。
// 在代码中实现：
//  1. 原子函数 atomicAdd(.) 执行数组的一次完整的读-写操作；
//  2. 传给 cudaMemcpy(.) 的主机内存可以是栈内存，也可以是堆内存；
//  3. 主机函数可以和设备函数同名，但要遵循重载原则（参数列表不一致）。
// === 原子函数：===
// 原子函数对其第一个参数指向的数据进行一次"读-写-改"的原子操作，是不可分割的操作。
// 第一个参数可以指向全局内存，也可以指向共享内存。
// 对所有参与的线程来说，原子操作是一个线程一个线程轮流进行的，没有明确的次序。
// 原子函数没有同步功能。
// 原子函数的返回值为所指地址的旧值。
//  - 加法：T atomicAdd(T *address, T val);
//  - 减法：T atomicSub(T *address, T val);
//  - 交换：T atomicExch(T *address, T val);
//  - 最小值：T atomicMin(T *address, T val);
//  - 最大值：T atomicMax(T *address, T val);
//  - 自增：T atomicInc(T *address, T val);
//  - 自减：T atomicDec(T *address, T val);
//  - 比较交换：T atomicCAS(T *address, T compare, T val);
//  === 邻居列表 ===
// 两个粒子互为邻居的判断：他们的距离不大于一个给定的截断距离 rc。
// 基本算法：对每一个给定的粒子，通过比较它与所有其他粒子的距离来判断相应粒子对是否互为邻居。

#include "error.cuh"
#include "floats.hpp"
#include "clock.cuh"

__global__ void reduce(real *x, real *y, const int N){
    int tid = threadIdx.x;
    int ind = tid + blockIdx.x * blockDim.x;

    extern __shared__ real curr_x[];
    curr_x[tid] = (ind < N) ? x[ind] : 0.0;

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2){
        if(tid < offset){
            curr_x[tid] += curr_x[tid + offset];
        }
        __syncthreads();
    }
    if(tid == 0){
        y[blockIdx.x] = curr_x[0];
    }
}

__global__ void reduce2(real *x, real *y, const int N){
    int tid = threadIdx.x;
    int ind = tid + blockIdx.x * blockDim.x;

    extern __shared__ real curr_x[];
    curr_x[tid] = (ind < N) ? x[ind] : 0.0;

    for (int offset = blockDim.x / 2; offset > 0; offset /=2){
        if(tid < offset){
            curr_x[tid] += curr_x[tid + offset];
        }
        __syncthreads();
    }

    if(tid == 0){
        atomicAdd(y, curr_x[0]);
    }
}

int main(){
    int N = 1e8;
    int M = N * sizeof(real);

    int bSize = 32;
    int gSize = (N + bSize - 1) / bSize;

    cout << FLOAT_PREC << endl;

    real *h_x, *h_y;
    h_x = new real[N];
    h_y = new real[gSize];
    for (int i = 0; i < N; ++i){
        h_x[i] = 1.23;
    }

    cudaClockStart

    real *d_x, *d_y;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMalloc(&d_y, gSize * sizeof(real)));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyDefault));

    cudaClockCurr

    reduce<<<gSize, bSize, (bSize+1)*sizeof(real)>>>(d_x, d_y, N);
    CHECK(cudaMemcpy(h_y, d_y, gSize * sizeof(real), cudaMemcpyDefault));
    real res = 0;
    for (int i = 0; i < gSize; ++i){
        res += h_y[i];
    }
    cout << "reduce result: " << res << endl;

    cudaClockCurr
    reduce<<<gSize, bSize, (bSize)*sizeof(real)>>>(d_x, d_y, N);
    CHECK(cudaMemcpy(h_y, d_y, gSize * sizeof(real), cudaMemcpyDefault));
    res = 0.0;
    for (int i = 0; i < gSize; ++i){
        res += h_y[i];
    }
    cout << "reduce result: " << res << endl;

    cudaClockCurr

    real *d_y2, *h_y2;
    h_y2 = new real(0.0);
    CHECK(cudaMalloc(&d_y2, sizeof(real)));

    reduce2<<<gSize, bSize, (bSize) * sizeof(real)>>>(d_x, d_y2, N);

    CHECK(cudaMemcpy(h_y2, d_y2, sizeof(real), cudaMemcpyDefault));
    cout << "reduce2 result: " << *h_y2 << endl;
    cudaClockCurr

    delete[] h_x;
    delete[] h_y;
    delete h_y2;
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_y2));

    return 0;
}
