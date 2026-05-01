#pragma once

#include <string>
#include <utility>

namespace dinov3_trt {

enum class StatusCode {
  kOk,
  kInvalidArgument,
  kNotFound,
  kUnavailable,
  kRuntimeError,
};

class Status {
 public:
  [[nodiscard]] static Status ok() { return Status(StatusCode::kOk, ""); }

  [[nodiscard]] static Status invalid_argument(std::string message) {
    return Status(StatusCode::kInvalidArgument, std::move(message));
  }

  [[nodiscard]] static Status not_found(std::string message) {
    return Status(StatusCode::kNotFound, std::move(message));
  }

  [[nodiscard]] static Status unavailable(std::string message) {
    return Status(StatusCode::kUnavailable, std::move(message));
  }

  [[nodiscard]] static Status runtime_error(std::string message) {
    return Status(StatusCode::kRuntimeError, std::move(message));
  }

  [[nodiscard]] bool is_ok() const noexcept { return code_ == StatusCode::kOk; }
  [[nodiscard]] StatusCode code() const noexcept { return code_; }
  [[nodiscard]] const std::string& message() const noexcept { return message_; }

 private:
  explicit Status(StatusCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  StatusCode code_;
  std::string message_;
};

}  // namespace dinov3_trt
