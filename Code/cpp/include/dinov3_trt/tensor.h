#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "dinov3_trt/preprocess.h"

namespace dinov3_trt {

inline constexpr int kPatchTokens = (kImageSize / kPatchSize) * (kImageSize / kPatchSize);
inline constexpr int kOutputTokens = 1 + kPatchTokens;
inline constexpr int kHiddenSize = 1024;
inline constexpr int kOutputCount = 4;
inline constexpr std::array<int, kOutputCount> kOutputLayerNumbers{4, 12, 16, 20};
inline constexpr std::array<const char*, kOutputCount> kOutputTensorNames{
    "feat_layer_4",
    "feat_layer_12",
    "feat_layer_16",
    "feat_layer_20",
};

enum class DataType {
  kFloat32,
  kFloat16,
  kInt8,
  kUInt8,
};

[[nodiscard]] constexpr std::size_t element_size_bytes(DataType data_type) noexcept {
  switch (data_type) {
    case DataType::kFloat32:
      return 4;
    case DataType::kFloat16:
      return 2;
    case DataType::kInt8:
    case DataType::kUInt8:
      return 1;
  }
  return 0;
}

struct TensorShape {
  std::array<std::int64_t, 4> dims{};
  std::size_t rank{0};

  [[nodiscard]] static constexpr TensorShape nchw(
      std::int64_t batch,
      std::int64_t channels,
      std::int64_t height,
      std::int64_t width) noexcept {
    return TensorShape{{batch, channels, height, width}, 4};
  }

  [[nodiscard]] static constexpr TensorShape btc(
      std::int64_t batch,
      std::int64_t tokens,
      std::int64_t channels) noexcept {
    return TensorShape{{batch, tokens, channels, 0}, 3};
  }

  [[nodiscard]] constexpr bool has_dynamic_dim() const noexcept {
    for (std::size_t index = 0; index < rank; ++index) {
      if (dims[index] < 0) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] constexpr std::int64_t element_count() const noexcept {
    if (rank == 0 || has_dynamic_dim()) {
      return 0;
    }

    std::int64_t count = 1;
    for (std::size_t index = 0; index < rank; ++index) {
      count *= dims[index];
    }
    return count;
  }
};

[[nodiscard]] constexpr TensorShape input_shape(std::int64_t batch) noexcept {
  return TensorShape::nchw(batch, kInputChannels, kImageSize, kImageSize);
}

[[nodiscard]] constexpr TensorShape output_shape(std::int64_t batch) noexcept {
  return TensorShape::btc(batch, kOutputTokens, kHiddenSize);
}

// Multi-resolution helpers. The DINOv3 ViT-L/16 contract keeps `kPatchSize`,
// `kInputChannels`, and `kHiddenSize` invariant; only the spatial extent and
// the resulting token count change. These helpers let the runtime tools
// support 224/336/518 (or any patch-aligned size) without revisiting the
// 224 default.
[[nodiscard]] constexpr int patch_tokens_for(int image_size) noexcept {
  const int grid = image_size / kPatchSize;
  return grid * grid;
}

[[nodiscard]] constexpr int output_tokens_for(int image_size) noexcept {
  return 1 + patch_tokens_for(image_size);
}

[[nodiscard]] constexpr TensorShape input_shape_for(
    std::int64_t batch,
    int image_size) noexcept {
  return TensorShape::nchw(batch, kInputChannels, image_size, image_size);
}

[[nodiscard]] constexpr TensorShape output_shape_for(
    std::int64_t batch,
    int image_size) noexcept {
  return TensorShape::btc(batch, output_tokens_for(image_size), kHiddenSize);
}

struct TensorView {
  void* data{nullptr};
  TensorShape shape{};
  DataType data_type{DataType::kFloat32};

  [[nodiscard]] constexpr std::size_t byte_size() const noexcept {
    const std::int64_t elements = shape.element_count();
    if (elements <= 0) {
      return 0;
    }
    return static_cast<std::size_t>(elements) * element_size_bytes(data_type);
  }

  [[nodiscard]] constexpr bool is_empty() const noexcept {
    return data == nullptr || byte_size() == 0;
  }
};

}  // namespace dinov3_trt
