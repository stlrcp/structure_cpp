#include <iostream>
// #include <cupy/complex.cuh>
#include "carray.cuh"
// #include <cupy/atomics.cuh>
// #include <cupy/math_constants.h>

typedef unsigned int O;
typedef bool T;


extern "C" __global__ void cupy_scan_naive(const CArray<unsigned int, 1, 1, 1> b, CArray<bool, 1, 1, 1> a, CArray<unsigned int, 1, 1, 1> out, CIndexer<1, 1> _ind) {

    __shared__ O smem1[512];
    __shared__ O smem2[32];
    const int n_warp = 512 / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
;
  #pragma unroll 1
  CUPY_FOR(i, _ind.size()) {
    _ind.set(i);

    O x = 0;
    if (i < a.size()) x = a[i];
    for (int j = 1; j < 32; j *= 2) {
        smem1[threadIdx.x] = x;  __syncwarp();
        if (lane_id - j >= 0) x += smem1[threadIdx.x - j];
        __syncwarp();
    }
    if (lane_id == 32 - 1) smem2[warp_id] = x;
    __syncthreads();
    if (warp_id == 0) {
        O y = 0;
        if (lane_id < n_warp) y = smem2[lane_id];
        for (int j = 1; j < n_warp; j *= 2) {
            smem2[lane_id] = y;  __syncwarp();
            if (lane_id - j >= 0) y += smem2[lane_id - j];
            __syncwarp();
        }
        smem2[lane_id] = y;
    }
    __syncthreads();
    if (warp_id > 0) x += smem2[warp_id - 1];
    int block_id = i / 512;
    if (block_id > 0) x += b[block_id - 1];
    
    if (i < a.size()) {
      printf("========================== x = %d ===== \n", x);
      out[i] = x;
    }
;
  }
  ;
}


int main(){
  O *h_b, *d_b;
  T *h_a, *d_a;
  O *h_out, *d_out;
  cudaMalloc(&d_b, 1 * sizeof(O));
  cudaMalloc(&d_a, 6 * sizeof(T));
  cudaMalloc(&d_out, 6 * sizeof(O));
  h_b = (O *)malloc(1 * sizeof(O));
  h_a = (T *)malloc(6 * sizeof(T));
  h_out = (O *)malloc(6 * sizeof(O));
  h_b[0] = 0;
  for (int i = 0; i < 6; i++){
    h_a[i] = false;
    h_out[i] = 1;
  }

  cudaMemcpy(d_b, h_b, 1*sizeof(O), cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, h_a, 6*sizeof(T), cudaMemcpyHostToDevice);

  int b_shape[1] = {1};
  int a_shape[1] = {6};
  int out_shape[1] = {6};
  int idx_shape[1] = {512};
  int *b_shape_d, *a_shape_d, *out_shape_d;
  cudaMalloc(&b_shape_d, 1*sizeof(int));
  cudaMalloc(&a_shape_d, 1*sizeof(int));
  cudaMalloc(&out_shape_d, 1*sizeof(int));

  cudaMemcpy(b_shape_d, b_shape, 1 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(a_shape_d, a_shape, 1 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(out_shape_d, out_shape, 1 * sizeof(int), cudaMemcpyHostToDevice);

  // const CArray<unsigned int, 1, 1, 1> input_b(d_b, b_shape_d);
  const CArray<unsigned int, 1, 1, 1> input_b(d_b, {1});
  // CArray<bool, 1, 1, 1> input_a(d_a, a_shape_d);
  CArray<bool, 1, 1, 1> input_a(d_a, {6});
  // CArray<unsigned int, 1, 1, 1> output(d_out, out_shape_d);
  CArray<unsigned int, 1, 1, 1> output(d_out, {6});
  dim3 grid(1, 1, 1);
  dim3 block(512, 1, 1);
  cupy_scan_naive<<<grid, block>>>(input_b, input_a, output, CIndexer<1, 1> (idx_shape));

  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out, 6*sizeof(O), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 6; i++){
    printf("================ h_out[%d] == %ld =\n", i, h_out[i]);
  }

  cudaFree(d_b);
  cudaFree(d_a);
  cudaFree(d_out);
  cudaFree(a_shape_d);
  cudaFree(b_shape_d);
  cudaFree(out_shape_d);
  return 0;
}
