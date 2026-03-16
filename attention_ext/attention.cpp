#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {

void check_attention_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v) {
  TORCH_CHECK(q.is_cuda(), "Q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "K must be a CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "V must be a CUDA tensor");
  TORCH_CHECK(q.scalar_type() == torch::kFloat32, "Q must be float32");
  TORCH_CHECK(k.scalar_type() == torch::kFloat32, "K must be float32");
  TORCH_CHECK(v.scalar_type() == torch::kFloat32, "V must be float32");
  TORCH_CHECK(q.is_contiguous(), "Q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "K must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "V must be contiguous");
  TORCH_CHECK(q.dim() == 2, "Q must have shape [L, d]");
  TORCH_CHECK(k.dim() == 2, "K must have shape [L, d]");
  TORCH_CHECK(v.dim() == 2, "V must have shape [L, d]");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "Q, K, V must share the same sequence length");
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1), "Q, K, V must share the same hidden size");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "Q, K, V must be on the same CUDA device");
}

}  // namespace

torch::Tensor attention_forward_naive_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v);

torch::Tensor attention_forward_tiled_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v);

torch::Tensor attention_forward_fused_softmax_pv_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v);

torch::Tensor attention_forward_naive(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v) {
  check_attention_inputs(q, k, v);
  c10::cuda::CUDAGuard device_guard(q.device());
  return attention_forward_naive_cuda(q, k, v);
}

torch::Tensor attention_forward_tiled(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v) {
  check_attention_inputs(q, k, v);
  c10::cuda::CUDAGuard device_guard(q.device());
  return attention_forward_tiled_cuda(q, k, v);
}

torch::Tensor attention_forward_fused_softmax_pv(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v) {
  check_attention_inputs(q, k, v);
  c10::cuda::CUDAGuard device_guard(q.device());
  return attention_forward_fused_softmax_pv_cuda(q, k, v);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attention_forward_naive", &attention_forward_naive, "Naive attention forward");
  m.def("attention_forward_tiled", &attention_forward_tiled, "Tiled attention forward");
  m.def("attention_forward_fused_softmax_pv", &attention_forward_fused_softmax_pv, "Fused softmax + P@V attention forward");
}
