// ====== 共享内存的合理使用 ======
// 共享内存是一种可以被程序员直接操作的缓存，主要作用有两个：
//  1. 减少核函数中对全局内存的访问次数，实现高效的线程块内部的通信；
//  2. 提高全局内存访问的合并度。
//  === 数组规约 ===
// 对于多线程程序，默认情况下不同线程的执行顺序是不固定的（线程间独立）。
// 采用 折半规约法，通过线程块对数据分片规约，最后再一并求和。
// 核函数中循环的每一轮都会被拆解、分配到线程块内的所有线程上执行，而不是一个线程连续执行一次完整循环。
// 核函数中代码是"单指令多线程"，代码真正的执行顺序与出现顺序可能不同。所以 线程0、1、... 127 之间实际上并行的。
// 保证一个线程块中所有线程在执行该语句后面的语句之前，都完全执行了前面的语句：
// 通过 __syncthreads() 实现一个线程块中所有线程按照代码出现的顺序执行指令，但是不同线程块之间依然是独立的、异步的。
// 共享内存变量，可以在核函数中通过限定符 __shared__ 定义一个共享内存变量，这样就相当于在每一个线程块中有一个该变量的副本。
// 虽然每个副本都是独立的，但核函数中对共享变量的操作都将同时作用在所有副本上。
// 核函数中可以直接使用函数外部由 #define 或 const 定义的常量，但在 MSVC 中限制了核函数使用 const 定义的常量。
// 利用共享内存进行线程块之间的合作（通信）之前，都要进行同步，以确保共享内存变量中数据对于所有线程块内的所有线程都是准备好的。
// 共享内存的生命周期仅在核函数内，所以必须在核函数结束前将共享内存中需要的结果保存到全局内存。
// 通过共享内存可以避免修改全局内存变量，同时不再要求全局内存数组为线程块大小的整数倍。
// 线程块的共享内存根据申请方式分为：静态共享内存变量和动态共享内存变量。
// 前者在核函数中定义共享内存大小（通过编译期常量），后者在主机调用核函数时指定大小（可以提高可维护性）。
//  === 矩阵转置 ===
// 由于共享内存访问速度快于全局内存，所以可以通过线程块内的共享内存将全局内存的非合并访问转为合并访问。
// 注意转置后的数组索引变换。
//  === 共享内存的 bank 冲突 ===
// 共享内存在物理上被分为 32个同样宽度（开普勒架构为8字节，其他为4字节）、能被同时访问的列向内存 bank。
// +++++++++++++++++++++++++++++
// bank0  bank1  ... bank31
// +++++++++++++++++++++++++++++
// layer1 layer1 ... layer1
// layer2 layer2 ... layer2
// ...
// layer32 layer32 ... layer32
// 只要同一个线程束内的多个线程不同时访问同一个 bank 中不同层的数据，该线程束对共享内存的访问就只需要一次内存事务。
// 当同一个线程束内的多个线程试图访问同一个 bank 中不同层的数据时，就会发生冲突。
// 在同一个线程束中的多个线程对同一个 bank 中的 n 层数据访问将导致 n 次内存事务。
// 称为发生了 n 路 bank 冲突。
// 当线程束内的 32 个线程同时访问同一个 bank 的32个不同层，这将导致 32路 bank 冲突。
// 对于非开普勒架构，每个共享内存的宽带为 4 字节；于是每一层的 32 个 bank 将对应 32 个 float 数组元素。
// 使用共享内存来改善全局内存的访问方式不一定会提高核函数的性能；
// 不要过早优化，在优化程序时要对不同的优化方案进行测试和比较。

#include "error.cuh"
#include "floats.hpp"
#include <iomanip>
#include <string>
#include <fstream>

#define TILE_DIM 32

__constant__ int c_TILE_DIM = 32;   // 设备内存中线程块中矩阵维度（线程块大小，最大1024）

void show(const real *matrix, const int N, std::string outfile, std::string title);
__global__ void transpose1(const real *src, real *dst, const int N);
__global__ void transpose2(const real *src, real *dst, const int N);
__global__ void transpose3(const real *src, real *dst, const int N);
__global__ void transpose4(const real *src, real *dst, const int N);

int main(){
    const int N = 500;
    const int M = N * N * sizeof(real);

    int SIZE = 0;
    CHECK(cudaMemcpyFromSymbol(&SIZE, c_TILE_DIM, sizeof(int)));

    const int grid_size_x = (N + SIZE - 1) / SIZE;
    const int grid_size_y = grid_size_x;

    const dim3 block_size(SIZE, SIZE);
    const dim3 grid_size(grid_size_x, grid_size_y);

    real *h_matrix_org, *h_matrix_res;
    h_matrix_org = new real[N * N];
    h_matrix_res = new real[N * N];
    for (int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j){
            h_matrix_org[i * N + j] = i * 1.0e-2;
        }
    }
    // show(h_matrix_org, N, "result.txt", "origin matrix");

    real *d_matrix_org, *d_matrix_res;
    CHECK(cudaMalloc(&d_matrix_org, M));
    CHECK(cudaMalloc(&d_matrix_res, M));
    CHECK(cudaMemcpy(d_matrix_org, h_matrix_org, M, cudaMemcpyDefault));

    float elapsed_time = 0;
    float curr_time = 0;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    cudaEventQuery(start);

    // 矩阵转置
    transpose1<<<grid_size, block_size>>>(d_matrix_org, d_matrix_res, N);
    CHECK(cudaMemcpy(h_matrix_res, d_matrix_res, M, cudaMemcpyDefault));
    // show(h_matrix_res, N, "result.txt", "transpose1");

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));
    printf("matrix transpose1 time cost: %f ms. \n", curr_time - elapsed_time);
    elapsed_time = curr_time;

    // 矩阵转置（全局内存非合并读取、合并写入）
    transpose2<<<grid_size, block_size>>>(d_matrix_org, d_matrix_res, N);
    CHECK(cudaMemcpy(h_matrix_res, d_matrix_res, M, cudaMemcpyDefault));
    // show(h_matrix_res, N, "matrix.txt", "transpose2");

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));
    printf("matrix transpose2 time cost: %f ms. \n", curr_time - elapsed_time);
    elapsed_time = curr_time;

    // 矩阵转置（通过共享内存全局内存合并读写）
    transpose3<<<grid_size, block_size>>>(d_matrix_org, d_matrix_res, N);
    CHECK(cudaMemcpy(h_matrix_res, d_matrix_res, M, cudaMemcpyDefault));
    //  show(h_matrix_res, N, "result.txt", "transpose3");

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));
    printf("matrix tranpose3 time cost: %f ms.\n", curr_time - elapsed_time);
    elapsed_time = curr_time;

    // 矩阵转置（通过共享内存、bank处理，实现全局内存合并读写）。
    transpose4<<<grid_size, block_size>>>(d_matrix_org, d_matrix_res, N);
    CHECK(cudaMemcpy(h_matrix_res, d_matrix_res, M, cudaMemcpyDefault));
    //  show(h_matrix_res, N, "result.txt", "transpose4");

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));
    printf("matrix transpose4 time cost: %f ms.\n", curr_time - elapsed_time);
    elapsed_time = curr_time;

    delete[] h_matrix_res;
    delete[] h_matrix_org;
    CHECK(cudaFree(d_matrix_org));
    CHECK(cudaFree(d_matrix_res));
    return 0;
}

void show(const real *x, const int N, std::string outfile, std::string title){
    std::fstream out(outfile, std::ios::app);
    if(!out.is_open()){
        std::cerr << "invalid output file: " << outfile << endl;
        return;
    }

    out << "\n\n-----------------" << title << endl;

    for (int i = 0; i < N; ++i){
        out << endl;
        for (int j = 0; j < N; ++j){
            out << std::setw(6) << x[i * N + j];
        }
    }
}

__global__ void transpose1(const real *src, real *dst, const int N){
    const int nx = threadIdx.x + blockIdx.x * c_TILE_DIM;
    const int ny = threadIdx.y + blockIdx.y * c_TILE_DIM;
    if(nx < N && ny < N){
        // 矩阵转置（合并读取、非合并写入）
        dst[nx * N + ny] = src[ny * N + nx];
    }
}

__global__ void transpose2(const real *src, real *dst, const int N){
    const int nx = threadIdx.x + blockIdx.x * c_TILE_DIM;
    const int ny = threadIdx.y + blockIdx.y * c_TILE_DIM;

    if(nx < N && ny < N){
        // 矩阵转置（非合并读取、合并写入）。
        dst[ny * N + nx] = __ldg(&src[nx * N + ny]);    // 显式调用 "__ldg()" 函数缓存全局内存
    }
}

__global__ void transpose3(const real *src, real *dst, const int N){
    __shared__ real s_mat[TILE_DIM][TILE_DIM];
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    int tx = threadIdx.x + bx;
    int ty = threadIdx.y + by;

    if(tx < N && ty < N){
        s_mat[threadIdx.y][threadIdx.x] = src[ty * N + tx];  // 全局内存中二维矩阵一维存储
    }
    __syncthreads();

    if(tx < N && ty < N){
        int x = by + threadIdx.x;
        int y = bx + threadIdx.y;
        dst[y * N + x] = s_mat[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose4(const real *src, real *dst, const int N){
    __shared__ real s_mat[TILE_DIM][TILE_DIM + 1];
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    int tx = threadIdx.x + bx;
    int ty = threadIdx.y + by;
    if(tx < N && ty < N){
        s_mat[threadIdx.y][threadIdx.x] = src[ty * N + tx];
    }
    __syncthreads();
    if(tx < N && ty < N){
        int x = by + threadIdx.x;
        int y = bx + threadIdx.y;
        dst[y * N + x] = s_mat[threadIdx.x][threadIdx.y];
    }
}
