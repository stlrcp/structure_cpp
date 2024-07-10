/*
// ==== CUDA 流 ====
// 一个CUDA流一般是指由主机发出的，在设备中执行的cuda操作序列（即和cuda有关的操作，如主机--设备数据传输和核函数执行）。
// 目前不考虑由设备段发出的流。
// 任何cuda的操作都存在与某个 cuda 流，要么是默认流（default stream），也称为空流；
// 要么是明确指定的流。非默认的 cuda 流（非空流）都是在主机端产生与销毁。
// 一个cuda流由类型为 cudaStream_t 的变量表示，创建与销毁的方式：
cudaStream_t stream;
CHECK(cudaStreamCreate(&stream));
...
CHECK(cudaStreamDestory(stream));
// 主机中可以产生多个相互独立的cuda流，并实现 cuda流之间的并行。
// 为了检查一个 cuda 流中所有操作是否都已在设备中执行完毕：
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
// cudaStreamSynchronize 会强制阻塞主机，直到其中的 stream 流执行完毕；
// cudaStreamQuery 不会阻塞主机，只是检查 cuda 流（stream）是否执行完毕，若是，则返回 cudaSuccess；否则，返回 cudaErrorNotReady。

// ==== 在默认流中重叠主机和设备计算 ====
// 同一个 cuda流在设备中都是顺序执行的. 在数组相加的例子中:
cudaMemcpy(d_x, h_x, M, cudaMemcpyDefault);
cudaMemcpy(d_y, h_y, M, cudaMemcpyDefault);
add<<<gridSize, blockSize>>>(d_x, d_y, d_z, N);
cudaMemcpy(h_z, d_z, M, cudaMemcpyDefault);
// 从设备的角度,以上4个cuda语句是按代码顺序执行的.
// 采用 cudaMemcpy 函数在主机和设备间拷贝数据,是具有隐式同步功能的.
// 所以从主机的角度看,数据传输是同步的或者说阻塞的,即主机在发出命令:
cudaMemcpy(d_x, h_x, M, cudaMemcpyDefault);
// 之后, 会等待该命令执行完毕,再接着往下走;数据传输时,主机是闲置的.
// 与此不同的是,核函数的启动是异步的或者说非阻塞的,即在主机发出命令:
add<<<gridSize, blockSize>>>(d_X, d_y, d_z, N);
// 之后,不会等待该命令执行完毕,而是立刻得到程序的控制权. 紧接着发出:
cudaMemcpy(h_z, d_z, M, cudaMemcpyDefault);
// 然而,该命令不会被立刻执行,因为其与核函数同处默认流,需要顺序执行.
// 所以,主机在发出核函数调用后会立刻发出下一个命令; 如果下一个命令是
// 主机中的某个计算任务,那么主机就会在设备执行核函数的同时执行计算.
// 这样就可以实现主机和设备的重叠计算.
// 当主机和设备的计算量相当时,将主机函数放在设备核函数后可以达到主机函数
// 与设备函数并发执行的效果,从而有效地隐藏主机函数的执行时间.

// ==== 非默认 cuda 流重叠多个核函数 ====
// 要实现多个核函数之间的并行必须使用多个非默认 cuda 流.
// 使用多个流相对于使用一个流有加速效果; 当流的数目超过某个阈值时,加速比就趋于饱和.
// 制约加速比的因素:
//   - GPU 计算资源,当核函数的线程总数超过某一值时,再增加流的数目就不会带来更高性能;
//   - GPU 中能够并发执行的核函数的上限.
// 指定核函数的cuda流的方法:
kernel_func<<<grid_size, block_size, 0, stream>>>(params);
// 在调用核函数时,如果不需要使用共享内存,则该项设为0; 同时指定 cuda 流的id.
// 计算能力为7.5 的GPU能执行的核函数上限值为128.

// ==== 非默认 cuda 流重叠核函数与数据传递 ====
// 要实现核函数执行与数据传输的并发(重叠), 必须让这两个操作处于不同的非默认流;
// 同时,数据传输需要使用 cudaMemcpy 的异步版本 cudaMemcpyAsync.
// 异步传输由GPU的 DMA (direct memory access) 实现,不需要主机的参与.
// 使用异步的数据传输函数时,需要将主机内存定义为不可分页内存或者固定内存,从而
// 防止在程序执行期间物理地址被修改. 如果将可分页内存传递给 cudaMemcpyAsync
// 则会导致同步传输.
// 主机不可分页内存的分配与释放:
cudaError_t cudaMallocHost(void **ptr, size_t size);
// 或者
cudaError_t cudaHostAlloc(void **ptr, size_t size);
cudaError_t cudaFreeHost(void *ptr);
// 要利用多个流提升性能,一种方法是将数据和相应计算操作分为若干等分,
// 然后在每个流中发布一个 cuda 操作序列.
// 如果核函数执行, 主机与设备间的数据传输这3个cuda操作能完全并行执行,
// 理论上最大加速比为 3.
*/


#include "error.cuh"
#include "floats.hpp"
#include <math.h>
#include <stdio.h>

const int NUM_REPEATS = 10;
const int N1 = 1024;
const int MAX_NUM_STREAMS = 30;
const int N2 = N1 * MAX_NUM_STREAMS;
const int M2 = sizeof(real) * N2;
cudaStream_t streams[MAX_NUM_STREAMS];   // cuda 流数组,全局变量由系统负责销毁.

const int N = 100000000;
const int M = sizeof(real) * N;
const int block_size = 128;
const int grid_size = (N-1) / block_size + 1;

void timing(const real *h_x, const real *h_y, real *h_z,
    const real *d_x, const real *d_y, real *d_z,
    const int ratio, bool overlap);
void timing(const real *d_x, const real *d_y, real *d_z,
    const int num);
void timing(const real *h_x, const real *h_y, real *h_z,
    real *d_x, real *d_y, real *d_z,
    const int num);

int main(void){
    real *h_x = (real*)malloc(M);
    real *h_y = (real*)malloc(M);
    real *h_z = (real*)malloc(M);
    for(int n=0; n<N; ++n){
        h_x[n] = 1.23;
        h_y[n] = 2.34;
    }
    real *d_x, *d_y, *d_z;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMalloc(&d_y, M));
    CHECK(cudaMalloc(&d_z, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

    // host and kernel overlap.
    printf("without CPU-GPU overlap (ratio = 10)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 10, false);
    printf("With CPU-GPU overlap (ratio = 10)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 10, true);

    printf("Without CPU-GPU overlap (ratio = 1)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 1, false);
    printf("With CPU-GPU overlap(ratio = 1)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 1, true);

    printf("Without CPU-GPU overlap (ratio = 1000)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 1000, false);
    printf("With CPU-GPU overlap (ratio = 1000)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 1000, true);

    // kernel and kernel overlap.
    for(int n=0; n<MAX_NUM_STREAMS; ++n){
        // 创建 cuda 流.
        CHECK(cudaStreamCreate(&(streams[n])));
    }

    for(int num=1; num <= MAX_NUM_STREAMS; ++num){
        timing(d_x, d_y, d_z, num);
    }

    for(int n=0; n<MAX_NUM_STREAMS; ++n){
        // 销毁 cuda 流
        CHECK(cudaStreamDestroy(streams[n]));
    }

    // kernel and data transfering overlap.
    real *h_x2, *h_y2, *h_z2;
    CHECK(cudaMallocHost(&h_x2, M));
    CHECK(cudaMallocHost(&h_y2, M));
    CHECK(cudaMallocHost(&h_z2, M));
    for(int n=0; n<N; ++n){
        h_x2[n] = 1.23;
        h_y2[n] = 2.34;
    }

    for(int i=0; i<MAX_NUM_STREAMS; i++){
        CHECK(cudaStreamCreate(&(streams[i])));
    }

    for(int num=1; num <= MAX_NUM_STREAMS; num *= 2){
        timing(h_x2, h_y2, h_z2, d_x, d_y, d_z, num);
    }

    for(int i=0; i<MAX_NUM_STREAMS; i++){
        CHECK(cudaStreamDestroy(streams[i]));
    }

    CHECK(cudaFreeHost(h_x2));
    CHECK(cudaFreeHost(h_y2));
    CHECK(cudaFreeHost(h_z2));
    free(h_x);
    free(h_y);
    free(h_z);
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));
    return 0;
}

void cpu_sum(const real *x, const real *y, real *z, const int N_host){
    for(int n=0; n<N_host; ++n){
        z[n] = x[n] + y[n];
    }
}

void __global__ gpu_sum(const real *x, const real *y, real *z){
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n < N){
        z[n] = x[n] + y[n];
    }
}

void timing(const real *h_x, const real *h_y, real *h_z,
        const real *d_x, const real *d_y, real *d_z,
        const int ratio, bool overlap){
    float t_sum = 0;
    float t2_sum = 0;

    for(int repeat =0; repeat <= NUM_REPEATS; ++repeat){
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        if(!overlap){
            cpu_sum(h_x, h_y, h_z, N/ratio);
        }
        gpu_sum<<<grid_size, block_size>>>(d_x, d_y, d_z);

        if(overlap){
            // 主机函数与设备核函数重叠
            cpu_sum(h_x, h_y, h_z, N/ratio);
        }

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        if(repeat > 0){
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);
}

void __global__ add(const real *d_x, const real *d_y, real *d_z){
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n < N1){
        for(int i=0; i<100000; ++i){
            d_z[n] = d_x[n] + d_y[n];
        }
    }
}

void timing(const real *d_x, const real *d_y, real *d_z, const int num){
    float t_sum = 0;
    float t2_sum =0;

    for(int repeat=0; repeat <= NUM_REPEATS; ++repeat){
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        for(int n=0; n<num; ++n){
            int offset = n * N1;
            // 指定各个核函数的 cuda 流,实现核函数的并行
            add<<<grid_size, block_size, 0, streams[n]>>>(d_x+offset, d_y+offset, d_z+offset);
        }

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

        if(repeat > 0){
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("%g\n", t_ave);
}

void __global__ add2(const real *x, const real *y, real *z, int N){
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n < N){
        for(int i=0; i< 40; ++i){
            z[n] = x[n] + y[n];
        }
    }
}

void timing(const real *h_x, const real *h_y, real *h_z,
        real *d_x, real *d_y, real *d_z,
        const int num){
    int N1 = N / num;
    int M1 = M / num;

    float t_sum = 0;
    float t2_sum = 0;

    for(int repeat =0; repeat <= NUM_REPEATS; ++repeat){
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);
        for (int i=0; i< num; i++){
            int offset  = i *N1;
            // 划分主机不可分页内存,实现异步的数据传输.
            // 每个 cuda 流都有各自的数据传输操作.
            CHECK(cudaMemcpyAsync(d_x+offset, h_x+offset, M1,
                cudaMemcpyHostToDevice, streams[i]));
            CHECK(cudaMemcpyAsync(d_y+offset, h_y+offset, M1,
                cudaMemcpyHostToDevice, streams[i]));

            int block_size = 128;
            int grid_size = (N1 - 1)/ block_size + 1;

            // 指定核函数的 cuda 流
            add2<<<grid_size, block_size, 0, streams[i]>>>(d_x+offset, d_y+offset, d_z+offset, N1);

            CHECK(cudaMemcpyAsync(h_z+offset, d_z+offset, M1,
                cudaMemcpyDeviceToHost, streams[i]));
        }

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

        if (repeat > 0){
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("%d %g\n", num, t_ave);
}
