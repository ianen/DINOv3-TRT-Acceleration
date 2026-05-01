#include <cmath>
#include <stdexcept>

#include "dinov3_trt/preprocess.h"
#include "dinov3_trt/status.h"
#include "dinov3_trt/tensor.h"
#include "dinov3_trt/trt_inferer.h"

namespace {

void expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void expect_close(float actual, float expected, float tolerance, const char* message) {
  if (std::fabs(actual - expected) > tolerance) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main() {
  using namespace dinov3_trt;

  static_assert(kOutputCount == 4, "DINOv3 TRT contract must expose four outputs");
  static_assert(kPatchTokens == 196, "224x224 ViT-L/16 should produce 14x14 patch tokens");
  static_assert(kOutputTokens == 197, "Output keeps CLS and drops register tokens");
  static_assert(kHiddenSize == 1024, "ViT-L/16 hidden size must remain 1024");
  static_assert(element_size_bytes(DataType::kFloat32) == 4, "FP32 byte width mismatch");
  static_assert(element_size_bytes(DataType::kFloat16) == 2, "FP16 byte width mismatch");

  const TensorShape input = input_shape(8);
  expect(input.rank == 4, "input must be NCHW");
  expect(input.dims[0] == 8, "input batch mismatch");
  expect(input.dims[1] == 3, "input channel mismatch");
  expect(input.dims[2] == 224, "input height mismatch");
  expect(input.dims[3] == 224, "input width mismatch");
  expect(input.element_count() == 8 * 3 * 224 * 224, "input element count mismatch");

  const TensorShape output = output_shape(8);
  expect(output.rank == 3, "output must be BTC");
  expect(output.dims[0] == 8, "output batch mismatch");
  expect(output.dims[1] == 197, "output token count mismatch");
  expect(output.dims[2] == 1024, "output hidden size mismatch");

  TensorView output_view{nullptr, output, DataType::kFloat32};
  expect(output_view.byte_size() == 8 * 197 * 1024 * 4, "output byte size mismatch");
  expect(output_view.is_empty(), "null output view should be empty");

  // Multi-resolution helpers must keep the legacy 224 numbers and produce the
  // expected 336 (442 tokens) and 518 (1025 tokens) shapes used by trtexec
  // engines built with --image-size 336 / 518.
  static_assert(patch_tokens_for(224) == 196, "224 grid must remain 14x14");
  static_assert(output_tokens_for(224) == 197, "224 must keep CLS+196 patches");
  static_assert(patch_tokens_for(336) == 21 * 21, "336 grid must be 21x21");
  static_assert(output_tokens_for(336) == 442, "336 must produce 1+441 tokens");
  static_assert(patch_tokens_for(518) == 1024, "518 grid must be 32x32");
  static_assert(output_tokens_for(518) == 1025, "518 must produce 1+1024 tokens");

  const TensorShape input_336 = input_shape_for(2, 336);
  expect(input_336.dims[0] == 2, "336 input batch mismatch");
  expect(input_336.dims[2] == 336 && input_336.dims[3] == 336,
         "336 input spatial mismatch");

  const TensorShape output_336 = output_shape_for(2, 336);
  expect(output_336.dims[0] == 2, "336 output batch mismatch");
  expect(output_336.dims[1] == 442, "336 output token count mismatch");
  expect(output_336.dims[2] == 1024, "336 output hidden mismatch");

  const TensorShape output_518 = output_shape_for(4, 518);
  expect(output_518.dims[0] == 4, "518 output batch mismatch");
  expect(output_518.dims[1] == 1025, "518 output token count mismatch");
  expect(output_518.dims[2] == 1024, "518 output hidden mismatch");

  const auto zero = normalize_rgb(0, 0, 0);
  expect_close(zero[0], -2.117904F, 1.0e-5F, "red normalization mismatch");
  expect_close(zero[1], -2.035714F, 1.0e-5F, "green normalization mismatch");
  expect_close(zero[2], -1.804444F, 1.0e-5F, "blue normalization mismatch");

  const Status ok = Status::ok();
  const Status error = Status::invalid_argument("bad input");
  expect(ok.is_ok(), "ok status mismatch");
  expect(!error.is_ok(), "error status mismatch");
  expect(error.code() == StatusCode::kInvalidArgument, "error code mismatch");
  expect(error.message() == "bad input", "error message mismatch");

  return 0;
}
