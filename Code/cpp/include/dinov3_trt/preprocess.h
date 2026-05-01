#pragma once

#include <array>
#include <cstdint>

namespace dinov3_trt {

inline constexpr int kImageSize = 224;
inline constexpr int kPatchSize = 16;
inline constexpr int kInputChannels = 3;
inline constexpr std::array<float, kInputChannels> kImageNetMean{0.485F, 0.456F, 0.406F};
inline constexpr std::array<float, kInputChannels> kImageNetStd{0.229F, 0.224F, 0.225F};

[[nodiscard]] constexpr std::array<float, kInputChannels> normalize_rgb(
    std::uint8_t red,
    std::uint8_t green,
    std::uint8_t blue) noexcept {
  return {
      ((static_cast<float>(red) / 255.0F) - kImageNetMean[0]) / kImageNetStd[0],
      ((static_cast<float>(green) / 255.0F) - kImageNetMean[1]) / kImageNetStd[1],
      ((static_cast<float>(blue) / 255.0F) - kImageNetMean[2]) / kImageNetStd[2],
  };
}

}  // namespace dinov3_trt
