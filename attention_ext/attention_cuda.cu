#include <torch/extension.h>

#include <cmath>
#include <cfloat>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kTile = 16;
constexpr int kSoftmaxThreads = 256;

void check_cuda_status() {
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}

// naive CUDA implementation kernel1
__global__ void qk_naive_kernel(
    const float* q,
    const float* k,
    float* scores,
    int L,
    int d,
    float inv_sqrt_d) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= L || col >= L) return;
  
  // score matrix S = Q @ K' / sqrt(d)
  float acc = 0.0f;
  for (int idx = 0; idx < d; ++idx) {
    acc += q[row * d + idx] * k[col * d + idx];
  }
  scores[row * L + col] = acc * inv_sqrt_d;
}

// naive CUDA implementation kernel2
__global__ void pv_naive_kernel(
    const float* probs,
    const float* v,
    float* out,
    int L,
    int d) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= L || col >= d) return;
  
  // P @ V
  float acc = 0.0f;
  for (int idx = 0; idx < L; ++idx) {
    acc += probs[row * L + idx] * v[idx * d + col];
  }
  out[row * d + col] = acc;
}

// naive CUDA implementation kernel3
__global__ void softmax_naive_kernel(
    const float* scores,
    float* probs,
    int L) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  extern __shared__ float shared[];

  float local_max = -FLT_MAX;
  for (int col = tid; col < L; col += blockDim.x) {
    local_max = fmaxf(local_max, scores[row * L + col]);
  }
  shared[tid] = local_max;
  // each thread puts its local maximum into shared memory
  __syncthreads();

  // tree reduction
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }

  const float row_max = shared[0];
  float local_sum = 0.0f;
  for (int col = tid; col < L; col += blockDim.x) {
    const float value = expf(scores[row * L + col] - row_max);
    probs[row * L + col] = value;
    local_sum += value;
  }
  shared[tid] = local_sum;
  __syncthreads();

  // sum reduction
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  const float row_sum = shared[0];
  for (int col = tid; col < L; col += blockDim.x) {
    probs[row * L + col] /= row_sum;
  }
}

__device__ __forceinline__ float warp_reduce_max(float value) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    // use warp shuffle intrinsics
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

// do tiled mat mul
__global__ void qk_tiled_kernel(
    const float* q,
    const float* k,
    float* scores,
    int L,
    int d,
    float inv_sqrt_d) {
  __shared__ float q_tile[kTile][kTile];
  __shared__ float k_tile[kTile][kTile];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int col = blockIdx.x * kTile + tx;
  const int row = blockIdx.y * kTile + ty;

  float acc = 0.0f;
  for (int tile_start = 0; tile_start < d; tile_start += kTile) {
    const int q_col = tile_start + tx;
    const int k_col = tile_start + ty;

    q_tile[ty][tx] = (row < L && q_col < d) ? q[row * d + q_col] : 0.0f;
    k_tile[ty][tx] = (col < L && k_col < d) ? k[col * d + k_col] : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int idx = 0; idx < kTile; ++idx) {
      acc += q_tile[ty][idx] * k_tile[idx][tx];
    }
    __syncthreads();
  }

  if (row < L && col < L) {
    scores[row * L + col] = acc * inv_sqrt_d;
  }
}

__global__ void pv_tiled_kernel(
    const float* probs,
    const float* v,
    float* out,
    int L,
    int d) {
  __shared__ float p_tile[kTile][kTile];
  __shared__ float v_tile[kTile][kTile];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int col = blockIdx.x * kTile + tx;
  const int row = blockIdx.y * kTile + ty;

  float acc = 0.0f;
  for (int tile_start = 0; tile_start < L; tile_start += kTile) {
    const int p_col = tile_start + tx;
    const int v_row = tile_start + ty;

    p_tile[ty][tx] = (row < L && p_col < L) ? probs[row * L + p_col] : 0.0f;
    v_tile[ty][tx] = (v_row < L && col < d) ? v[v_row * d + col] : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int idx = 0; idx < kTile; ++idx) {
      acc += p_tile[ty][idx] * v_tile[idx][tx];
    }
    __syncthreads();
  }

  if (row < L && col < d) {
    out[row * d + col] = acc;
  }
}

__global__ void softmax_warp_kernel(
    const float* scores,
    float* probs,
    int L) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & (warpSize - 1);
  const int warp = tid / warpSize;
  const int num_warps = blockDim.x / warpSize;

  __shared__ float warp_partials[32];

  float local_max = -FLT_MAX;
  for (int col = tid; col < L; col += blockDim.x) {
    local_max = fmaxf(local_max, scores[row * L + col]);
  }
  local_max = warp_reduce_max(local_max);
  if (lane == 0) {
    warp_partials[warp] = local_max;
  }
  __syncthreads();

  if (warp == 0) {
    float value = (lane < num_warps) ? warp_partials[lane] : -FLT_MAX;
    value = warp_reduce_max(value);
    if (lane == 0) {
      warp_partials[0] = value;
    }
  }
  __syncthreads();
  const float row_max = warp_partials[0];

  float local_sum = 0.0f;
  for (int col = tid; col < L; col += blockDim.x) {
    const float value = expf(scores[row * L + col] - row_max);
    probs[row * L + col] = value;
    local_sum += value;
  }
  local_sum = warp_reduce_sum(local_sum);
  if (lane == 0) {
    warp_partials[warp] = local_sum;
  }
  __syncthreads();

  if (warp == 0) {
    float value = (lane < num_warps) ? warp_partials[lane] : 0.0f;
    value = warp_reduce_sum(value);
    if (lane == 0) {
      warp_partials[0] = value;
    }
  }
  __syncthreads();
  const float row_sum = warp_partials[0];

  for (int col = tid; col < L; col += blockDim.x) {
    probs[row * L + col] /= row_sum;
  }
}

__global__ void softmax_pv_fused_kernel(
    const float* scores,
    const float* v,
    float* out,
    int L,
    int d) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & (warpSize - 1);
  const int warp = tid / warpSize;
  const int num_warps = blockDim.x / warpSize;

  __shared__ float warp_partials[32];
  __shared__ float weight_tile[kTile];

  float local_max = -FLT_MAX;
  for (int col = tid; col < L; col += blockDim.x) {
    local_max = fmaxf(local_max, scores[row * L + col]);
  }
  local_max = warp_reduce_max(local_max);
  if (lane == 0) {
    warp_partials[warp] = local_max;
  }
  __syncthreads();

  if (warp == 0) {
    float value = (lane < num_warps) ? warp_partials[lane] : -FLT_MAX;
    value = warp_reduce_max(value);
    if (lane == 0) {
      warp_partials[0] = value;
    }
  }
  __syncthreads();
  const float row_max = warp_partials[0];

  float local_sum = 0.0f;
  for (int col = tid; col < L; col += blockDim.x) {
    local_sum += expf(scores[row * L + col] - row_max);
  }
  local_sum = warp_reduce_sum(local_sum);
  if (lane == 0) {
    warp_partials[warp] = local_sum;
  }
  __syncthreads();

  if (warp == 0) {
    float value = (lane < num_warps) ? warp_partials[lane] : 0.0f;
    value = warp_reduce_sum(value);
    if (lane == 0) {
      warp_partials[0] = value;
    }
  }
  __syncthreads();
  const float inv_row_sum = 1.0f / warp_partials[0];

  // do softmax and P@V together in a block, not sending P to global memory
  for (int out_col = tid; out_col < d; out_col += blockDim.x) {
    float acc = 0.0f;
    for (int col_base = 0; col_base < L; col_base += kTile) {
      if (tid < kTile) {
        const int col = col_base + tid;
        weight_tile[tid] = (col < L) ? expf(scores[row * L + col] - row_max) * inv_row_sum : 0.0f;
      }
      __syncthreads();

      const int tile_width = min(kTile, L - col_base);
      for (int tile_col = 0; tile_col < tile_width; ++tile_col) {
        acc += weight_tile[tile_col] * v[(col_base + tile_col) * d + out_col];
      }
      __syncthreads();
    }
    out[row * d + out_col] = acc;
  }
}

}  // namespace

torch::Tensor attention_forward_naive_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v) {
  const int L = static_cast<int>(q.size(0));
  const int d = static_cast<int>(q.size(1));
  const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));

  auto scores = torch::empty({L, L}, q.options());
  auto probs = torch::empty({L, L}, q.options());
  auto out = torch::empty({L, d}, q.options());

  dim3 mm_block(kTile, kTile);
  dim3 qk_grid((L + kTile - 1) / kTile, (L + kTile - 1) / kTile);
  dim3 pv_grid((d + kTile - 1) / kTile, (L + kTile - 1) / kTile);

  qk_naive_kernel<<<qk_grid, mm_block>>>(
      q.data_ptr<float>(),
      k.data_ptr<float>(),
      scores.data_ptr<float>(),
      L,
      d,
      inv_sqrt_d);
  check_cuda_status();

  softmax_naive_kernel<<<L, kSoftmaxThreads, kSoftmaxThreads * sizeof(float)>>>(
      scores.data_ptr<float>(),
      probs.data_ptr<float>(),
      L);
  check_cuda_status();

  pv_naive_kernel<<<pv_grid, mm_block>>>(
      probs.data_ptr<float>(),
      v.data_ptr<float>(),
      out.data_ptr<float>(),
      L,
      d);
  check_cuda_status();

  return out;
}

torch::Tensor attention_forward_tiled_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v) {
  const int L = static_cast<int>(q.size(0));
  const int d = static_cast<int>(q.size(1));
  const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));

  auto scores = torch::empty({L, L}, q.options());
  auto probs = torch::empty({L, L}, q.options());
  auto out = torch::empty({L, d}, q.options());

  dim3 mm_block(kTile, kTile);
  dim3 qk_grid((L + kTile - 1) / kTile, (L + kTile - 1) / kTile);
  dim3 pv_grid((d + kTile - 1) / kTile, (L + kTile - 1) / kTile);

  qk_tiled_kernel<<<qk_grid, mm_block>>>(
      q.data_ptr<float>(),
      k.data_ptr<float>(),
      scores.data_ptr<float>(),
      L,
      d,
      inv_sqrt_d);
  check_cuda_status();

  softmax_warp_kernel<<<L, kSoftmaxThreads>>>(
      scores.data_ptr<float>(),
      probs.data_ptr<float>(),
      L);
  check_cuda_status();

  pv_tiled_kernel<<<pv_grid, mm_block>>>(
      probs.data_ptr<float>(),
      v.data_ptr<float>(),
      out.data_ptr<float>(),
      L,
      d);
  check_cuda_status();

  return out;
}

torch::Tensor attention_forward_fused_softmax_pv_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v) {
  const int L = static_cast<int>(q.size(0));
  const int d = static_cast<int>(q.size(1));
  const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));

  auto scores = torch::empty({L, L}, q.options());
  auto out = torch::empty({L, d}, q.options());

  dim3 mm_block(kTile, kTile);
  dim3 qk_grid((L + kTile - 1) / kTile, (L + kTile - 1) / kTile);

  qk_tiled_kernel<<<qk_grid, mm_block>>>(
      q.data_ptr<float>(),
      k.data_ptr<float>(),
      scores.data_ptr<float>(),
      L,
      d,
      inv_sqrt_d);
  check_cuda_status();

  softmax_pv_fused_kernel<<<L, kSoftmaxThreads>>>(
      scores.data_ptr<float>(),
      v.data_ptr<float>(),
      out.data_ptr<float>(),
      L,
      d);
  check_cuda_status();

  return out;
}
