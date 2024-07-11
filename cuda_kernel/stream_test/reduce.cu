/*
//  ====== 线程束基本函数与协作组 ======
// 线程束（warp），即一个线程块中连续 32 个线程
// === 单指令-多线程模式 ===
// 一个GPU被分成若干个流多处理器（SM）。核函数中定义的线程块（block）在执行时
// 将被分配到还没有完全占满的 SM。
// 一个 block 不会被分配到不同的 SM，同时一个 SM 中可以有多个 block。不同的
// block 之间可以并发也可以顺序执行，一般不能同步。
// 当某些 block 完成计算任务后，对应的 SM 会部分或完全空闲，然后会有新的
// block 被分配到空闲SM。

// 一个 SM 以32个线程（warp）为单位产生、管理、调度、执行线程。
// 一个 SM 可以处理多个 block，一个 block 可以分为若干个 warp。

// 在同一时刻，一个warp中的线程只能执行一个共同的指令或者闲置，即单指令-多线程执行模型，
// （single instruction multiple thread, SIMT）.

// 当一个线程束中线程顺序的执行判断语句中的不同分支时，称为发生了分支发散（branch divergence）。
if (condition){
    A;
} else {
    B;
}
// 首先，满足 condition 的线程或执行语句A，其他的线程会闲置；
// 然后，不满足条件的将会执行语句B，其他线程闲置。
// 当语句A和B的指令数差不多时，整个 warp 的执行效率就比没有分支的情况 低一半。
// 一般应当在核函数中尽量避免分支发散，但有时这也是不可避免的。
// 如数组计算中常用的判断语句：
if(n<N){
    // do something.
}
// 该分支判断最多影响最后一个 block 中的某些 warp 发生分支发散，一般不会显著地影响性能。
// 有时能通过 合并判断语句 的方式减少分支发散；另外，如果两分支中有一个分支不包含指令，
// 则即使发生分支分散也不会显著影响性能。
// 注意不同架构中的线程调度机制

//  ==== 线程束内的线程同步函数 ====
// __syncwarp(.): 当所涉及的线程都在一个线程束内时，可以将线程块同步函数 __syncthreads() 换成
// 一个更加廉价的线程束同步函数。
// __syncwarp(.): 简称 束内同步函数。
// 函数参数是一个代表掩码的无符号整型数，默认值是全部32个二进制位都为1，代表线程束中
// 的所有线程都参与同步。
// 关于掩码（mask）的简介文章：https://zhuanlan.zhihu.com/p/352025616

// ==== 更多线程束内的基本函数 ====
// == 线程束表决函数 ==
//  - unsgined __ballot_sync(unsigned mask, int predicate),
//    如果线程束内第 n个线程参与计算（旧掩码）且 predicate 值非零，则返回的无符号整型数（新掩码）
//    的第 n 个二进制位为1，否则为0；
//  - int __all_sync(unsigned mask, int predicate),
//    线程束内所有参与线程的 predicate 值均非零，则返回1，否则返回0；
//  - int __any_sync(unsigned mask, int predicate),
//    线程束内所有参与线程的 predicate 值存在非零，则返回1，否则返回0.
// == 线程束洗牌函数：
//  - T __shfl_sync(unsigned mask, T v, int srcLane, int w = warpSize),
//    参与线程返回标号为 srcLane 的线程中变量 v 的值。
//    该函数将一个线程中的数据广播到所有线程
//  - T __shfl_up_sync(unsigned mask, T v, unsigned d, int w=warpSize),
//    标号为t的参与线程返回标号为 t-d 的线程中变量 v 的值， t-d<0 的线程返回 t 线程的变量v。
//    该函数是一种将数据向上平移的操作，即将低线程号的值平移到高线程号。
//    例如当 w=8、d=2时，2-7号线程将返回 0-5线程中变量 v 的值；0-1号线程返回自己的 v。
//  - T __shfl_down_sync(unsigned mask, T v, unsigned d, int w=warpSize),
//    标号为 t 的参与线程返回标号为 t+d 的线程中变量v的值，t+d>w 的线程返回t线程的变量 v。
//    该函数是一种将数据向下平移的操作，即将高线程号的值平移到低线程号。
//    例如当 w=8、d=2时，0-5号线程将返回 2-7 号线程中变量 v 的值，6-7号线程将返回自己的 v。
//  - T __shfl__xor_sync(unsigned mask, T v, int laneMask, int w=warpSize),
//    标号为 t 的参与线程返回标号为 t^laneMask 的线程中变量 v 的值。
//    该函数让线程束内的线程两两交换数据。
// 每个线程束洗牌函数都有一个可选参数 w，默认是线程束大小（32），且只能取2，4，8，16，32.
// 当 w 小于 32 时，相当于逻辑上的线程束大小是 w，其他规则不变。
// 此时，可以定义一个束内索引：（假设使用一维线程块）
int laneId = threadIdx.x % w;   // 线程索引与束内索引的对应关系
// 假设线程块大小为16， w为8：
//   线程索引：0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
//   束内索引：0 1 2 3 4 5 6 7 0 1  2  3  4  5  6  7
// 参数中的 mask 称为掩码，是一个无符号整型，具有32位，一般用十六进制表示：
const unsigned FULL_MASK = 0xffffffff;  // '0x'表示十六进制数； '0b'表示二进制数。
// 或者
#define FULL_MASK 0xffffffff
// 以上所有线程束内函数都有 _sync 后缀，表示这些函数都具有 隐式的同步功能。


//  ====  协作组  ====
// 协作组（cooperative groups），可以看作是线程块和线程束同步机制的推广，
// 提供包括线程块内部的同步与协作、线程块之间（网格级）的同步与协作、以及
// 设备与设备之间的同步与协作。
// 使用协作组需要包含如下头文件：
#include <cooperative_groups.h>
using namespace cooperative_groups;
//  === 线程块级别的协作组 ===
// 协作组编程模型中最基本的类型是线程组 thread_group, 其包含如下成员：
//  - void sync(), 同步组内所有线程；
//  - unsigned size(), 返回组内总的线程数目，即组的大小；
//  - unsigned thread_rank(), 返回当前调用该函数的线程在组内的标号（从0计数）；
//  - bool is_valid(), 如果定义的组违反了任何 cuda 限制，返回 false， 否则 true；
// 线程组类型有一个导出类型，线程块 thread_block, 其中定义了额外的函数：
//  - dim3 group_index(), 返回当前调用该函数的线程的线程块指标，等价于 blockIdx；
//  - dim3 thread_index(), 返回当前调用该函数的线程的线程指标，等价于 threadIdx；
// 通过 this_thread_block() 初始化一个线程块对象：
thread_block g = this_thread_block();   // 相当于一个线程块类型的常量。
// 此时：
g.sync() <===>  __syncthreads()
g.group_index() <===> blockIdx
g.thread_index() <===> threadIdx
// 通过 tiled_partition() 可以将一个线程块划分为若干片（tile），每一片构成一个新的线程组。
// 目前，仅支持将片的大小设置为 2 的整数次方且不大于 32.
thread_group g32 = tiled_partition(this_thread_block(), 32);  // 将线程块划分为线程束。
// 可以继续将线程组划分为更细的线程组：
thread_group g4 = tiled_partition(g32, 4);
// 采用模板、在编译期划分 线程块片（thread block tile）:
thread_block_tile<32> g32 = tiled_partition<32>(this_thread_block());
thread_block_tile<32> g4 = tiled_partition<4>(this_thread_block());
// 线程块片具有额外的函数（类似线程束内函数）：
//   unsigned ballot(int predicate);
//   int all(int predicate);
//   int any(int predicate);
//   T shfl(T v, int srcLane);
//   T shfl_up(T v, unsigned d);
//   T shfl_down(T v, unsigned d);
//   T shfl_xor(T v, unsigned d);
// 与一般的线程束不同，线程组内的所有线程都要参与代码运算计算；
// 同时，线程组内函数不需要指定宽度，因为该宽度就是线程块片的大小。

// ===== 数组规约程序的进一步优化 =====
//  === 提高线程利用率 ===
// 在当前的规约程序中，当 offset=64，只用了 1/2 的线程；当 offset=32，只用了 1/4 的线程；...
// 最终，当 offset=1，只用了 1/128 的线程；
// 规约过程一共用了 log2(128) = 7步，平均线程利用率（1/2 + 1/4 + ... + 1/128）/ 7 => 1/7

// 而在规约前的数据拷贝中线程利用率为 100%，可以尽量把计算放在在规约前：让一个线程处理多个数据。

// 一个线程处理相邻若干个数据会导致全局内存的非合并访问。要保证全局内存的合并访问，这里需要
// 保证相邻线程处理相邻数据，一个线程访问的数据需要有某种跨度。
// 该跨度可以是线程块的大小，也可以是网格的大小；对于一维情况，分别是 blockDim.x 和 blockDim.x * gridDim.x

//  === 避免反复分配与释放设备内存 ===
// 设备内存的分配与释放是比较耗时的。
// 通过采用静态全局内存替代动态全局内存，实现编译期的设备内存分配可以更加高效。
// 此外，应当尽量避免在较内存循环反复的分配和释放设备内存
*/


#include "error.cuh"
#include "floats.hpp"
#include "clock.cuh"
#include <cooperative_groups.h>
// #include <iostream>
using namespace cooperative_groups;

__constant__ unsigned FULL_MASK = 0xffffffff;
#define __gSize 10240
__device__ real static_y[__gSize];

__global__ void reduce_syncthreads(real *x, real *y, const int N);
__global__ void reduce_syncwarp(real *x, real *y, const int N);
__global__ void reduce_shfl_down(real *x, real *y, const int N);
__global__ void reduce_cp(real *x, real *y, const int N);
__global__ void reduce_cp_grid(const real *x, real *y, const int N);
real reduce_wrap(const real *x, const int N, const int gSize, const int bSize);
real reduce_wrap_static(const real *x, const int N, const int gSize, const int bSize);

int main(){
    int N = 1e8;
    int M = N * sizeof(real);

    int bSize = 32;
    int gSize = (N + bSize - 1) / bSize;

    cout << FLOAT_PREC << endl;

    real *h_x, *h_x2, *h_y, *h_y2, *h_res;
    h_x = new real[N];
    h_x2 = new real[N];
    h_y = new real[gSize];
    h_y2 = new real[gSize];
    h_res = new real(0.0);
    for (int i = 0; i < N; ++i){
        h_x[i] = 1.23;
        h_x2[i] = 1.23;
    }
    real initRes = 0.0;
    for (int i = 0; i < gSize; ++i){
        h_y2[i] = 0.0;
    }

    cudaClockStart

    real *d_x, *d_y, *d_res;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMalloc(&d_y, gSize * sizeof(real)));
    CHECK(cudaMalloc(&d_res, sizeof(real)));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyDefault));

    cudaClockCurr

    reduce_syncthreads<<<gSize, bSize, (bSize)*sizeof(real)>>>(d_x, d_y, N);

    CHECK(cudaMemcpy(h_y, d_y, gSize * sizeof(real), cudaMemcpyDefault));
    real res = 0;
    for (int i = 0; i < gSize; ++i){
        res += h_y[i];
    }
    cout << "reduce_syncthreads result: " << res << endl;
    cudaClockCurr

    CHECK(cudaMemcpy(d_res, &initRes, sizeof(real), cudaMemcpyDefault));
    reduce_syncwarp<<<gSize, bSize, bSize * sizeof(real)>>>(d_x, d_res, N);
    CHECK(cudaMemcpy(h_res, d_res, sizeof(real), cudaMemcpyDefault));
    cout << "reduce_syncwrap result: " << *h_res << endl;
    cudaClockCurr

    CHECK(cudaMemcpy(d_res, &initRes, sizeof(real), cudaMemcpyDefault));
    reduce_shfl_down<<<gSize, bSize, bSize * sizeof(real)>>>(d_x, d_res, N);
    CHECK(cudaMemcpy(h_res, d_res, sizeof(real), cudaMemcpyDefault));
    cout << "reduce_shfl_down result: " << *h_res << endl;
    cudaClockCurr

    CHECK(cudaMemcpy(d_res, &initRes, sizeof(real), cudaMemcpyDefault));
    reduce_cp<<<gSize, bSize, bSize * sizeof(real)>>>(d_x, d_res, N);
    CHECK(cudaMemcpy(h_res, d_res, sizeof(real), cudaMemcpyDefault));
    cout << "reduce_cp result: " << *h_res << endl;
    cudaClockCurr

    reduce_cp_grid<<<gSize, bSize, bSize*sizeof(real)>>>(d_x, d_y, N);
    CHECK(cudaMemcpy(h_y, d_y, gSize * sizeof(real), cudaMemcpyDefault));
    res = 0.0;
    for (int i = 0; i < gSize; ++i){
        res += h_y[i];
    }
    cout << "reduce_cp_grid result: " << res << endl;
    cudaClockCurr

    res = reduce_wrap(d_x, N, 10240, 128);
    cout << "reduce_wrap result: " << res << endl;
    cudaClockCurr

    res = reduce_wrap_static(d_x, N, 10240, 128);
    cout << "reduce_wrap_static result: " << res << endl;
    cudaClockCurr

    delete[] h_x;
    delete[] h_y;
    delete h_res;
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_res));
    return 0;
}

__global__ void reduce_syncthreads(real *x, real *y, const int N){
    int tid = threadIdx.x;   // 线程块中线程在 x 方向的 id
    int ind = tid + blockIdx.x * blockDim.x;

    extern __shared__ real block_x[];
    block_x[tid] = (ind < N) ? x[ind] : 0;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2){
        if(tid < offset){
            block_x[tid] += block_x[tid + offset];
        }
        __syncthreads();
    }

    if(tid == 0){
        y[blockIdx.x] = block_x[0];
    }
}

__global__ void reduce_syncwarp(real *x, real *y, const int N){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int ind = bid * blockDim.x + tid;

    extern __shared__ real block_arr[];
    block_arr[tid] = (ind < N) ? x[ind] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset >= 32; offset /= 2){
        if(tid < offset){
            block_arr[tid] += block_arr[tid + offset];
        }
        __syncthreads();
    }

    for (int offset = 16; offset > 0; offset /= 2){
        if(tid < offset){
            block_arr[tid] += block_arr[tid + offset];
        }
        __syncwarp();
    }

    if(tid == 0){
        atomicAdd(y, block_arr[0]);
    }
}

__global__ void reduce_shfl_down(real *x, real *y, const int N){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int ind = bid * blockDim.x + tid;

    extern __shared__ real block_arr[];
    block_arr[tid] = (ind < N) ? x[ind] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset >= 32; offset /= 2){
        if(tid < offset){
            block_arr[tid] += block_arr[tid + offset];
        }
        __syncthreads();
    }

    // 在线程寄存器上定义一个变量y
    real curr_y = block_arr[tid];

    for (int offset = 16; offset > 0; offset /= 2){
        curr_y += __shfl_down_sync(FULL_MASK, curr_y, offset);
    }

    if(tid == 0){
        atomicAdd(y, curr_y);
    }
}

__global__ void reduce_cp(real *x, real *y, const int N){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int ind = bid * blockDim.x + tid;

    extern __shared__ real block_arr[];
    block_arr[tid] = (ind < N) ? x[ind] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset >= 32; offset /= 2){
        if(tid < offset){
            block_arr[tid] += block_arr[tid + offset];
        }
        __syncthreads();
    }

    real curr_y = block_arr[tid];

    thread_block_tile<32> g32 = tiled_partition<32>(this_thread_block());

    for (int offset = 16; offset > 0; offset /= 2){
        curr_y += g32.shfl_down(curr_y, offset);
    }

    if(tid == 0){
        atomicAdd(y, curr_y);
    }
}

__global__ void reduce_cp_grid(const real *x, real *y, const int N){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    extern __shared__ real block_arr[];

    real curr_y = 0.0;

    const int stride = blockDim.x * gridDim.x;
    for (int n = bid * blockDim.x + tid; n < N; n+= stride){
        curr_y += x[n];
    }

    block_arr[tid] = curr_y;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset >= 32; offset /= 2){
        if(tid < offset){
            block_arr[tid] += block_arr[tid + offset];
        }
        __syncthreads();
    }

    curr_y = block_arr[tid];
    thread_block_tile<32> g32 = tiled_partition<32>(this_thread_block());
    for (int offset = 16; offset > 0; offset /= 2){
        curr_y += g32.shfl_down(curr_y, offset);
    }

    if(tid == 0){
        y[bid] = curr_y;
    }
}

real reduce_wrap(const real *x, const int N, const int gSize, const int bSize){
    const int ymem = gSize * sizeof(real);
    const int smem = bSize * sizeof(real);

    real h_y[1] = {0};
    real *d_y;
    CHECK(cudaMalloc(&d_y, ymem));

    reduce_cp_grid<<<gSize, bSize, smem>>>(x, d_y, N);
    reduce_cp_grid<<<1, 1024, 1024 * sizeof(real)>>>(d_y, d_y, gSize);

    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDefault));
    CHECK(cudaFree(d_y));

    return h_y[0];
}

real reduce_wrap_static(const real *x, const int N, const int gSize, const int bSize){
    real *d_y;
    CHECK(cudaGetSymbolAddress((void **)&d_y, static_y));
    reduce_cp_grid<<<gSize, bSize, bSize * sizeof(real)>>>(x, d_y, N);
    reduce_cp_grid<<<1, 1024, 1024 * sizeof(real)>>>(d_y, d_y, gSize);

    real h_y[1] = {0};
    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDefault));

    return h_y[0];
}
