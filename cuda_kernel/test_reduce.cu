/*
#include "error.cuh"
#include <stdlib.h>

int main(int argc, char *argv[])
{
	int device_id = 0;
	if (argc > 1) device_id = atoi(argv[1]);

	CHECK(cudaSetDevice(device_id));

	cudaDeviceProp prop;
	CHECK(cudaGetDeviceProperties(&prop, device_id));

	printf("Device id: %d \n", device_id);
	printf("Device name: %s\n", prop.name);
	printf("Compute capability: %d.%d\n", prop.major, prop.minor);
	printf("Amount of global memory: %g GB\n", prop.totalGlobalMem/(1024.0*1024*1024));
	printf("Amount of constant memory: %g KB\n", prop.totalConstMem/1024.0);
	printf("Maximum grid size: %d, %d, %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("Maximum block size: %d, %d, %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("Number of SMs: %d\n", prop.multiProcessorCount);
	printf("Maximum amount of shared memory per block: %g KB \n", prop.sharedMemPerBlock/1024.0);
	printf("Maximum amount of shared memory per SM； %g KB \n", prop.sharedMemPerMultiprocessor/1024.0);
	printf("Maximum number of registers per block: %d K\n", prop.regsPerBlock/1024);
	printf("Maximum number of registers per SM: %d K\n", prop.regsPerMultiprocessor/1024);
	printf("Maximum number of threads per block: %d \n", prop.maxThreadsPerBlock);
	printf("Maximum number of threads per SM: %d \n", prop.maxThreadsPerMultiProcessor);

	return 0;
}
*/


#include <stdlib.h>
#include "error.cuh"
#include <sys/time.h>

void initDevice(int devNum)
{
	int dev = devNum;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("using device %d: %s \n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));
}

void initialData_int(int* ip, int size)
{
	time_t t;
	srand((unsigned)time(&t));
	for (int i=0; i<size; i++)
	{
		ip[i] = int(rand()&0xff);
	}
}

double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}

__global__ void reduceNeighbored(int* g_idata, int* g_odata, unsigned int n)
{
	// set thread ID
	unsigned int tid = threadIdx.x;
	// boundary check
	if (tid >= n) return;
	// convert global data pointer to the
	int *idata = g_idata + blockIdx.x * blockDim.x;
	// in_place reduction in global memory
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if ((tid % (2 * stride)) == 0)
		{
			idata[tid] += idata[tid + stride];
		}
		// synchronize within block
		__syncthreads();
	}
	// write reault for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}



int main(int argc, char** argv)
{
	initDevice(0);

	// initialization

	int size = 1 << 24;
	printf(" with array size %d ", size);

	// execution configuration
	int blocksize = 1024;
	if (argc > 1)
	{
		blocksize = atoi(argv[1]);    // 从命令行输入设置block大小
	}
	dim3 block(blocksize, 1);
	dim3 grid((size -1) / block.x + 1, 1);
	printf("grid %d block %d \n", grid.x, block.x);

	// allocate host memory
	size_t bytes = size * sizeof(int);
	int *idata_host = (int*)malloc(bytes);
	int *odata_host = (int*)malloc(grid.x * sizeof(int));
	int *tmp = (int*)malloc(bytes);

	// initialize the array
	initialData_int(idata_host, size);

	memcpy(tmp, idata_host, bytes);
	double timeStart, timeElaps;
	int gpu_sum = 0;

	// device memory
	int *idata_dev = NULL;
	int * odata_dev = NULL;
	CHECK(cudaMalloc((void**)&idata_dev, bytes));
	CHECK(cudaMalloc((void**)&odata_dev, grid.x * sizeof(int)));

	// cpu reduction 对照组
	int cpu_sum = 0;
	timeStart = cpuSecond();
	// cpu_sum = recursiveReduce(tmp, size);
	for (int i=0; i<size; i++)
		cpu_sum += tmp[i];
	timeElaps = 1000 * (cpuSecond() - timeStart);

	printf("cpu sum: %d \n", cpu_sum);
	printf("cpu reduction elapsed %lf ms cpu_sum: %d\n", timeElaps, cpu_sum);

	// kernel reduceNeighbored
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	timeStart = cpuSecond();
	reduceNeighbored<<<grid, block>>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i=0; i<grid.x; i++)
		gpu_sum += odata_host[i];
	timeElaps = 1000 * (cpuSecond() - timeStart);

	printf("gpu sum: %d \n", gpu_sum);
	printf("gpu reduceNeighbored elapsed %lf ms  <<<grid %d, block %d>>>\n", timeElaps, grid.x, block.x);

	// free host memory
	free(idata_host);
	free(odata_host);
	CHECK(cudaFree(idata_dev));
	CHECK(cudaFree(odata_dev));

	// reset device
	cudaDeviceReset();

	// check the results
	if (gpu_sum == cpu_sum)
	{
		printf("Test success! \n");
	}
	return EXIT_SUCCESS;
}
