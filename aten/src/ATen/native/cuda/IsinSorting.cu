#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/TensorMeta.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/TensorCompare.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/isin_native.h>
#include <ATen/ops/result_type.h>
#endif

#include <cmath>

namespace at::native {

namespace {

template <typename scalar_t>
__global__ void isin_sorting_kernel(
    const scalar_t* __restrict__ sorted_test,
    int64_t num_test_elements,
    const scalar_t* __restrict__ elements,
    int64_t num_elements,
    bool invert,
    bool* __restrict__ output) {
  CUDA_KERNEL_LOOP_TYPE(i, num_elements, int64_t) {
    const scalar_t val = elements[i];
    int64_t start = 0;
    int64_t end = num_test_elements;
    while (start < end) {
      const int64_t mid = start + ((end - start) >> 1);
      if (sorted_test[mid] < val) {
        start = mid + 1;
      } else {
        end = mid;
      }
    }
    const bool match =
        (start < num_test_elements) && (sorted_test[start] == val);
    output[i] = invert ? !match : match;
  }
}

void isin_sorting_cuda(
    const Tensor& elements,
    const Tensor& test_elements,
    bool invert,
    const Tensor& out) {
  const ScalarType common_dtype = at::result_type(elements, test_elements);
  Tensor sorted_test =
      std::get<0>(test_elements.to(common_dtype).ravel().sort());
  Tensor elements_flat = elements.to(common_dtype).contiguous().view(-1);

  const int64_t num_elements = elements_flat.numel();
  const int64_t num_test_elements = sorted_test.numel();

  Tensor output = out.contiguous();

  constexpr int block_size = 256;
  const int num_blocks = std::min(
      static_cast<int>(ceil_div<int64_t>(num_elements, block_size)),
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 4);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_V2(
      common_dtype,
      "isin_sorting_cuda",
      AT_WRAP([&] {
        isin_sorting_kernel<scalar_t><<<num_blocks, block_size, 0, stream>>>(
            sorted_test.const_data_ptr<scalar_t>(),
            num_test_elements,
            elements_flat.const_data_ptr<scalar_t>(),
            num_elements,
            invert,
            output.mutable_data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      AT_EXPAND(AT_ALL_TYPES),
      kHalf,
      kBFloat16);

  if (!out.is_contiguous()) {
    out.copy_(output);
  }
}

} // anonymous namespace

TORCH_IMPL_FUNC(isin_Tensor_Tensor_out_cuda)
(const Tensor& elements,
 const Tensor& test_elements,
 bool /*assume_unique*/,
 bool invert,
 const Tensor& out) {
  if (elements.numel() == 0) {
    return;
  }

  // Brute force when there are few elements to look up, else sort the test set
  // and binary search per element. Unlike the CPU concat-sort, this kernel's
  // crossover gates on elements.numel(): sorting amortizes the test set across
  // many lookups, so it wins once elements is large for a given test set size.
  if (elements.numel() <=
      static_cast<int64_t>(
          175.0 * std::pow(static_cast<double>(test_elements.numel()), 0.155))) {
    out.fill_(invert);
    isin_default_stub(kCUDA, elements, test_elements, invert, out);
  } else {
    isin_sorting_cuda(elements, test_elements, invert, out);
  }
}

} // namespace at::native
