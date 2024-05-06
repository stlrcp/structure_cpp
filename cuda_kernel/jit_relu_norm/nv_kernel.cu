template <typename T, int N>
struct Tensor {
  __device__ T& operator[](int ind) {
    return data[ind];
  };

  T* data;
  int size[N];
  int stride[N];
};


#define NVFUSER_DEFINE_MAGIC_ZERO          \
  __shared__ int nvfuser_zero_s;           \
  if (threadIdx.x == 0)                    \
    nvfuser_zero_s = 0;                    \
  __syncthreads();                         \
  atomicMin(&nvfuser_zero_s, threadIdx.x); \
  int nvfuser_zero = nvfuser_zero_s;


#define NVFUSER_UPDATE_MAGIC_ZERO \
  do {                            \
    nvfuser_zero <<= 1;           \
  } while (0);


__device__ float reciprocal(float x) {
  return 1 / x;
}


__device__ double relu(double x) {
  return x <= 0 ? 0 : x;
}

__device__ float relu(float x) {
  return x <= 0 ? 0 : x;
}

__device__ float relu(int64_t x) {
  return x <= 0 ? 0 : x;
}

__device__ float relu(int x) {
  return x <= 0 ? 0 : x;
}



namespace index_utils {

// Utility functions

// Total size of provided dimension
template <typename _dim3>
__device__ __forceinline__ int size(const _dim3& d) {
  return (int)d.x * (int)d.y * (int)d.z;
}

// Linearized indexing of idx based on dim, if bool==false that dimension does
// not participate
template <bool X, bool Y, bool Z, typename _dim3, typename _dim3_2>
__device__ int maskedOffset(const _dim3& idx, const _dim3_2& dim) {
  int offset = 0;
  if (Z)
    offset += idx.z;
  if (Y)
    offset = offset * dim.y + idx.y;
  if (X)
    offset = offset * dim.x + idx.x;
  return offset;
}

// Linearized indexing of idx based on dim. All dimensions participate.
template <typename _dim3, typename _dim3_2>
__device__ int offset(const _dim3& idx, const _dim3_2& dim) {
  int offset = idx.z;
  offset = offset * dim.y + idx.y;
  offset = offset * dim.x + idx.x;
  return offset;
}

// Masks the provided dim3, those == false get truncated to 1
template <bool X, bool Y, bool Z, typename _dim3>
__device__ dim3 maskedDims(const _dim3& dim) {
  return dim3{
      X ? (unsigned)dim.x : 1U,
      Y ? (unsigned)dim.y : 1U,
      Z ? (unsigned)dim.z : 1U};
}

// Provides total size of dim with masking, those dims == false do not
// participate in the size calculation
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK, typename _dim3>
__device__ int maskedSize(const _dim3& dim) {
  return size(maskedDims<X_BLOCK, Y_BLOCK, Z_BLOCK>(dim));
}

// Checks if provided idx is zero on those dims == true
template <bool X, bool Y, bool Z, typename _dim3>
__device__ bool maskedIsZero(const _dim3& idx) {
  bool isZero = true;
  if (X)
    isZero = isZero && idx.x == 0;
  if (Y)
    isZero = isZero && idx.y == 0;
  if (Z)
    isZero = isZero && idx.z == 0;
  return isZero;
}

// Checks if provided idx is zero on those dims == true
template <bool X, bool Y, bool Z, typename _dim3, typename _dim3_2>
__device__ bool maskedIsLast(const _dim3& idx, const _dim3_2& dim) {
  bool isZero = true;
  if (X)
    isZero = isZero && idx.x == dim.x - 1;
  if (Y)
    isZero = isZero && idx.y == dim.y - 1;
  if (Z)
    isZero = isZero && idx.z == dim.z - 1;
  return isZero;
}

} // namespace index_utils

// Default block synchronization. Just use __barrier_sync
namespace block_sync {

__forceinline__ __device__ void init() {}

// Thread-block synchronization
__forceinline__ __device__ void sync() {
  __syncthreads();
}

} // namespace block_sync



template <bool X_REDUCE, bool Y_REDUCE, bool Z_REDUCE, typename T, typename Func, typename _dim3, typename _dim3_2>
__device__ void blockReduce( T& out, const T& inp_val, Func reduction_op, const _dim3& thread_idx,
    const _dim3_2& block_dim, T* shared_mem, bool read_pred, bool write_pred, T init_val) {
  // If this thread will output a final result
  bool should_write = index_utils::maskedIsZero<X_REDUCE, Y_REDUCE, Z_REDUCE>(thread_idx);

  // Size of the reduction segments
  unsigned int reduction_size = index_utils::maskedSize<X_REDUCE, Y_REDUCE, Z_REDUCE>(block_dim);

  // Index into the reduction segment
  unsigned int reduction_tid = index_utils::maskedOffset<X_REDUCE, Y_REDUCE, Z_REDUCE>(thread_idx, block_dim);

  // Index of the reduction segment
  unsigned int reduction_idx = index_utils::maskedOffset<!X_REDUCE, !Y_REDUCE, !Z_REDUCE>(thread_idx, block_dim);

  // Offset into smem for the current thread
  unsigned int smem_offset = reduction_idx * reduction_size + reduction_tid;

  // Initialize shared memory
  if (read_pred) {
    shared_mem[smem_offset] = inp_val;
  } else {
    shared_mem[smem_offset] = init_val;
  }

  block_sync::sync();
  // Reduce down to nearest power of 2 for the tree reduction:
  int np2 = 1 << (31 - __clz(reduction_size));

  if (reduction_tid < np2 && reduction_tid + np2 < reduction_size) {
    reduction_op(shared_mem[smem_offset], shared_mem[smem_offset + np2]);
  }
  block_sync::sync();

  // loop peel the final iteration to save one syncthread for the end
  for (int factor = np2 / 2; factor > 1; factor >>= 1) {
    if (reduction_tid < factor) {
      reduction_op(shared_mem[smem_offset], shared_mem[smem_offset + factor]);
    }
    block_sync::sync();
  }

  if (should_write && write_pred) {
    T result = out;
    reduction_op(result, shared_mem[smem_offset]);
    if (reduction_size > 1) {
      reduction_op(result, shared_mem[smem_offset + 1]);
    }
    out = result;
  }
  block_sync::sync();
}



// Use the same pred for both reads and writes
template <
    bool X_REDUCE,
    bool Y_REDUCE,
    bool Z_REDUCE,
    typename T,
    typename Func,
    typename _dim3,
    typename _dim3_2>
__device__ void blockReduce(
    T& out,
    const T& inp_val,
    Func reduction_op,
    const _dim3& thread_idx,
    const _dim3_2& block_dim,
    T* shared_mem,
    bool read_write_pred,
    T init_val) {
  blockReduce<X_REDUCE, Y_REDUCE, Z_REDUCE, T, Func, _dim3, _dim3_2>(
      out,
      inp_val,
      reduction_op,
      thread_idx,
      block_dim,
      shared_mem,
      read_write_pred,
      read_write_pred,
      init_val);
}



namespace broadcast {
// Broadcasts within partitioned groups of threads.
//
// X_THREAD: Broadcast from threadIdx.x == 0 if true
// Y_THREAD: Broadcast from threadIdx.y == 0 if true
// Z_THREAD: Broadcast from threadIdx.z == 0 if true
// inp_val: Per-thread source value. Only valid when the thread is a source.
// out: Per-thread output location
//
template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD, typename T>
__device__ void blockBroadcast(
    T& out,
    const T& inp_val,
    T* shared_mem,
    bool read_write_pred) {
  const bool has_valid_data = (!X_THREAD || threadIdx.x == 0) &&
      (!Y_THREAD || threadIdx.y == 0) && (!Z_THREAD || threadIdx.z == 0);

  const auto shared_offset =
      index_utils::maskedOffset<!X_THREAD, !Y_THREAD, !Z_THREAD>(
          threadIdx, blockDim);

  if (has_valid_data && read_write_pred) {
    shared_mem[shared_offset] = inp_val;
  }

  block_sync::sync();

  if (read_write_pred) {
    out = shared_mem[shared_offset];
  }

  block_sync::sync();
}

} // namespace broadcast





__global__ void kernel1(Tensor<float, 4> T0, int64_t i4, int64_t i3, Tensor<float, 4> T26, Tensor<float, 2> T6, Tensor<float, 2> T32) {
  alignas(16) extern __shared__ char array[];
  void* shared_mem = array;
  NVFUSER_DEFINE_MAGIC_ZERO
  printf("==================== i4 = %lld, i3 = %lld ==========\n", i4, i3);
  double d99;
  d99 = (double)(i3);
  double d100;
  d100 = 1.00000000000000000e+00 * d99;
  double d101;
  d101 = (double)(i4);
  double d102;
  d102 = d100 * d101;
  double d14;
  d14 = (double)(i3);
  double d15;
  d15 = 1.00000000000000000e+00 * d14;
  double d16;
  d16 = (double)(i4);
  double d17;
  d17 = d15 * d16;
  double d69;
  d69 = reciprocal(d17);
  if ((((((int)blockIdx.x) * 4) + 3) < (T0.size[0] * T0.size[1]))) {
    float T29[((4 * 1) * 1)];
    #pragma unroll
    for(int i232 = 0; i232 < 4; ++i232) {
      T29[i232] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(int i232 = 0; i232 < 4; ++i232) {
      T29[i232]
         = T0[(((((int)blockIdx.x) * 4) + (i232 + nvfuser_zero)) * (i4 * i3)) + ((int)threadIdx.x)];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    float T31[4];
    float T33[4];
    // Alias Allocation - register
    auto& T30 = T29;
    #pragma unroll
    for(int i248 = 0; i248 < 4; ++i248) {
      int i679;
      i679 = (((int)blockIdx.x) * 4) + (i248 + nvfuser_zero);
      float T27[1];
      T27[0] = 0.00000000000000000e+00;
      float T34[1];
      T34[0]
         = T29[i248];
      blockReduce<true, false, false>(
        T27[0],
        T34[0],
        [](float &a, float b) { a = a + b; },
        threadIdx,
        blockDim,
        static_cast<float*>(shared_mem),
        (((((int)blockIdx.x) * 4) + (i248 + nvfuser_zero)) < (T0.size[0] * T0.size[1])),
        float(0.00000000000000000e+00));
      float T3[1];
      if ((((int)threadIdx.x) == 0)) {
        T3[0]
          = T27[0]
          / (float) d102;
      }
      if ((((int)threadIdx.x) == 0)) {
        T31[i248]
          = T3[0]
          * (float) 1.00000001490116119e-01;
      }
      float T19[1];
      broadcast::blockBroadcast<true, false, false>(
        T19[0],
        T3[0],
        static_cast<float*>(shared_mem),
        true);
      float T20[(1 * 1)];
      T20[0]
        = T29[i248]
        - T19[0];
      float T4[1];
      T4[0] = 0.00000000000000000e+00;
      float T35[1];
      float T28[1];
      T28[0]
        = T20[0]
        * T20[0];
      T35[0]
         = T28[0];
      blockReduce<true, false, false>(
        T4[0],
        T35[0],
        [](float &a, float b) { a = a + b; },
        threadIdx,
        blockDim,
        static_cast<float*>(shared_mem),
        (((((int)blockIdx.x) * 4) + (i248 + nvfuser_zero)) < (T0.size[0] * T0.size[1])),
        float(0.00000000000000000e+00));
      if ((((int)threadIdx.x) == 0)) {
        T33[i248]
           = T4[0];
      }
      float T21[1];
      if ((((int)threadIdx.x) == 0)) {
        T21[0]
          = T4[0]
          * (float) d69;
      }
      float T22[1];
      if ((((int)threadIdx.x) == 0)) {
        T22[0]
          = T21[0]
          + (float) 9.99999974737875164e-06;
      }
      float T23[1];
      if ((((int)threadIdx.x) == 0)) {
        T23[0]
           = rsqrtf(T22[0]);
      }
      float T24[1];
      broadcast::blockBroadcast<true, false, false>(
        T24[0],
        T23[0],
        static_cast<float*>(shared_mem),
        true);
      float T25[1];
      T25[0]
        = T20[0]
        * T24[0];
      T30[i248]
         = relu(T25[0]);
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(int i244 = 0; i244 < 4; ++i244) {
      T26[(((((int)blockIdx.x) * 4) + (i244 + nvfuser_zero)) * (i4 * i3)) + ((int)threadIdx.x)]
         = T30[i244];
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(int i247 = 0; i247 < 4; ++i247) {
      if ((((int)threadIdx.x) == 0)) {
        T6[((((int)blockIdx.x) * 4) + (i247 + nvfuser_zero))]
           = T31[i247];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(int i250 = 0; i250 < 4; ++i250) {
      if ((((int)threadIdx.x) == 0)) {
        T32[((((int)blockIdx.x) * 4) + (i250 + nvfuser_zero))]
           = T33[i250];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  } else {
    float T29[((4 * 1) * 1)];
    #pragma unroll
    for(int i232 = 0; i232 < 4; ++i232) {
      T29[i232] = 0;
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(int i232 = 0; i232 < 4; ++i232) {
      int i398;
      i398 = (((int)blockIdx.x) * 4) + (i232 + nvfuser_zero);
      if ((i398 < (T0.size[0] * T0.size[1]))) {
        T29[i232]
           = T0[(i398 * (i4 * i3)) + ((int)threadIdx.x)];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    float T31[4];
    float T33[4];
    // Alias Allocation - register
    auto& T30 = T29;
    #pragma unroll
    for(int i248 = 0; i248 < 4; ++i248) {
      int i757;
      i757 = (((int)blockIdx.x) * 4) + (i248 + nvfuser_zero);
      float T27[1];
      T27[0] = 0.00000000000000000e+00;
      float T34[1];
      T34[0]
         = T29[i248];
      blockReduce<true, false, false>(
        T27[0],
        T34[0],
        [](float &a, float b) { a = a + b; },
        threadIdx,
        blockDim,
        static_cast<float*>(shared_mem),
        (i757 < (T0.size[0] * T0.size[1])),
        float(0.00000000000000000e+00));
      float T3[1];
      T3[0]
        = T27[0]
        / (float) d102;
      T31[i248]
        = T3[0]
        * (float) 1.00000001490116119e-01;
      float T19[1];
      broadcast::blockBroadcast<true, false, false>(
        T19[0],
        T3[0],
        static_cast<float*>(shared_mem),
        true);
      float T20[(1 * 1)];
      T20[0]
        = T29[i248]
        - T19[0];
      float T4[1];
      T4[0] = 0.00000000000000000e+00;
      float T35[1];
      float T28[1];
      T28[0]
        = T20[0]
        * T20[0];
      T35[0]
         = T28[0];
      blockReduce<true, false, false>(
        T4[0],
        T35[0],
        [](float &a, float b) { a = a + b; },
        threadIdx,
        blockDim,
        static_cast<float*>(shared_mem),
        (i757 < (T0.size[0] * T0.size[1])),
        float(0.00000000000000000e+00));
      T33[i248]
         = T4[0];
      float T21[1];
      T21[0]
        = T4[0]
        * (float) d69;
      float T22[1];
      T22[0]
        = T21[0]
        + (float) 9.99999974737875164e-06;
      float T23[1];
      T23[0]
         = rsqrtf(T22[0]);
      float T24[1];
      broadcast::blockBroadcast<true, false, false>(
        T24[0],
        T23[0],
        static_cast<float*>(shared_mem),
        true);
      float T25[1];
      T25[0]
        = T20[0]
        * T24[0];
      T30[i248]
         = relu(T25[0]);
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(int i244 = 0; i244 < 4; ++i244) {
      int i477;
      i477 = (((int)blockIdx.x) * 4) + (i244 + nvfuser_zero);
      if ((i477 < (T0.size[0] * T0.size[1]))) {
        T26[(i477 * (i4 * i3)) + ((int)threadIdx.x)]
           = T30[i244];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(int i247 = 0; i247 < 4; ++i247) {
      int i490;
      i490 = (((int)blockIdx.x) * 4) + (i247 + nvfuser_zero);
      if (((i490 < (T0.size[0] * T0.size[1])) && (((int)threadIdx.x) == 0))) {
        T6[i490]
           = T31[i247];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
    #pragma unroll
    for(int i250 = 0; i250 < 4; ++i250) {
      int i500;
      i500 = (((int)blockIdx.x) * 4) + (i250 + nvfuser_zero);
      if (((i500 < (T0.size[0] * T0.size[1])) && (((int)threadIdx.x) == 0))) {
        T32[i500]
           = T33[i250];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO
  }
}
