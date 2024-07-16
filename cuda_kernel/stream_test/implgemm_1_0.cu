#include <cuda_runtime.h>
#include <stdio.h>
#include "error.cuh"
// #include "conv2d.h"

typedef struct{
    float *input;            // 输入数据地址
    float *weight;           // 权值数据地址
    float *bias;             // 偏置值数据地址
    float *output;           // 输出数据地址
    unsigned int n;          // batch size
    unsigned int c;          // channel number
    unsigned int h;          // 数据高
    unsigned int w;          // 数据宽
    unsigned int k;          // 卷积核数量
    unsigned int r;          // 卷积核高
    unsigned int s;          // 卷积核宽
    unsigned int u;          // 卷积在高方向上的步长
    unsigned int v;          // 卷积在宽方向上的步长
    unsigned int p;          // 卷积在高方向上的补边
    unsigned int q;          // 卷积在宽方向上的补边
    unsigned int Oh;         // 卷积结果高
    unsigned int Ow;         // 卷积结果宽
} param_t;

// 合并 KRS 维度
__global__ void
implgemm(param_t param)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    if(x >= param.Oh * param.Ow || y >= param.k || z >= param.n){
        return;
    }
    // 当前线程处理的数据点在 oh、ow 上的坐标
    int posOh = x / param.Ow;
    int posOw = x % param.Ow;

    int posh_ori = posOh * param.u - param.p;
    int posw_ori = posOw * param.v - param.q;

    float sum = 0.0;

    int inOffset = z * param.c * param.h * param.w;
    int weiOffset = y * param.c * param.r * param.s;
    int inChannelOffset = param.h * param.w;
    int weightChannelOffset = param.r * param.s;

    for (int i = 0; i < param.r * param.s * param.c; i++){
        int curC = i / (param.r * param.s);
        int curR = (i % (param.r * param.s)) / param.s;
        int curS = (i % (param.r * param.s)) % param.s;
        int curH = posh_ori + curR;
        int curW = posw_ori + curS;
        int weiOffsetTmp = curC * weightChannelOffset + curR * param.s + curS;
        int inOffsetTmp = curC * inChannelOffset + curH * param.w + curW;
        if(curH >= 0 && curW >= 0 && curW < param.w && curH < param.h){
            sum += param.weight[weiOffset + weiOffsetTmp] * param.input[inOffset + inOffsetTmp];
        }
    }
    //  计算输出偏移
    int outOffset = z * param.k * param.Oh * param.Ow + y * param.Oh * param.Ow + x;
    param.output[outOffset] = sum;
}

void launch_implgemm(param_t param){
    unsigned int n = param.n;
    unsigned int c = param.c;
    unsigned int h = param.h;
    unsigned int w = param.w;
    unsigned int k = param.k;
    unsigned int r = param.r;
    unsigned int s = param.s;
    unsigned int u = param.u;
    unsigned int v = param.v;
    unsigned int p = param.p;
    unsigned int q = param.q;

    int outh = (h - r + 2 * p) / u + 1;
    int outw = (w - s + 2 * q) / v + 1;

    int blockx = ((outh * outw + 15) / 16);   // blockx number
    int blocky = (k + 15) / 16;              // blocky number
    int blockz = n;
    int threadx = 16;               // threadx number per block
    int thready = 16;               // thready number per block
    int threadz = 1;                // threadz number per block
    dim3 block(threadx, thready, threadz);
    dim3 grid(blockx, blocky, blockz);
    implgemm<<<grid, block>>>(param);
}

int main(int argc, char **argv){
    unsigned int n = atoi(argv[1]);
    unsigned int c = atoi(argv[2]);
    unsigned int h = atoi(argv[3]);
    unsigned int w = atoi(argv[4]);
    unsigned int k = atoi(argv[5]);
    unsigned int r = atoi(argv[6]);
    unsigned int s = atoi(argv[7]);
    unsigned int u = atoi(argv[8]);
    unsigned int v = atoi(argv[9]);
    unsigned int p = atoi(argv[10]);
    unsigned int q = atoi(argv[11]);

    int outh = (h - r + 2 * p) / u + 1;
    int outw = (w - s + 2 * q) / v + 1;
    double M = k;
    double N = n * outh * outw;
    double K = c * r * s;
    double temp = n * outh * outw * 1e-9f;
    double flopsPerConv = temp * M * K * 2.0;
    float *input = (float *)malloc(n * c * h * w * sizeof(float));
    float *weight = (float *)malloc(k * c * r * s * sizeof(float));
    float *bias = (float *)malloc(k * sizeof(float));
    float *output = (float *)malloc(n * k * outh * outw * sizeof(float));
    float *output_host = (float *)malloc(n * k * outh * outw * sizeof(float));

    float *input_device, *weight_device, *bias_device, *output_device;
    cudaMalloc((void **)&input_device, n * c * h * w * sizeof(float));
    cudaMalloc((void **)&weight_device, k * c * r * s * sizeof(float));
    cudaMalloc((void **)&bias_device, k * sizeof(float));
    cudaMalloc((void **)&output_device, n * k * outh * outw * sizeof(float));

    for (int i = 0; i < n * c * h * w; i++){
        input[i] = (rand() % 255) / 255.0;
    }
    for (int i = 0; i<k*c*r*s; i++){
        weight[i] = (rand() % 255) / 255.0;
    }
    for (int i = 0; i < k; i++){
        bias[i] = (rand() % 255) / 255.0;
    }
    for (int i = 0; i < n * k * outh * outw; i++){
        output[i] = 0.0;
        output_host[i] = 0.0;
    }

    cudaMemcpy(input_device, input, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_device, weight, k * c * r * s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_device, bias, k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output_device, output, n * k * outh * outw * sizeof(float), cudaMemcpyHostToDevice);

    // Convolution parameter
    param_t param;

    param.input = input_device;
    param.weight = weight_device;
    param.bias = bias_device;
    param.output = output_device;
    param.n = n;
    param.c = c;
    param.h = h;
    param.w = w;
    param.k = k;
    param.r = r;
    param.s = s;
    param.u = u;
    param.v = v;
    param.p = p;
    param.q = q;
    param.Oh = outh;
    param.Ow = outw;

    //  ====== warm up and get result ======
    launch_implgemm(param);
    cudaMemcpy(output_host, output_device, n * k * outh * outw * sizeof(float), cudaMemcpyDeviceToHost);

    //  ====== cost time test ======
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    float time_elapsed = 0.0;

    int iternum = 10;
    for (int i = 0; i < iternum; i++){
        launch_implgemm(param);
    }
    cudaEventRecord(stop, 0);

    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&time_elapsed, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float timePerConv = time_elapsed / iternum;
    double gflops = flopsPerConv / (timePerConv / 1000.0f);
    printf("%2d %2d %2d %2d %d %d %2d\n", n, h, w, c, r, s, k);
    printf("time: %f ms\n", timePerConv);
    printf("Performance: %f GFlops\n", gflops);

    cudaFree(input_device);
    cudaFree(weight_device);
    cudaFree(output_device);
    free(input);
    free(weight);
    free(output);
    free(output_host);
    return 0;
}
