// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#ifdef FA_ONLY_HDIM128
  #define HEADDIM_SWITCH(HEADDIM, ...)        \
    [&] {                                     \
      if (HEADDIM == 128) {                   \
        constexpr static int kHeadDim = 128;  \
        return __VA_ARGS__();                 \
      } else {                                \
        throw std::runtime_error("fa2-turing was compiled for head_dim=128 only"); \
      }                                       \
    }()
#else
  #define HEADDIM_SWITCH(HEADDIM, ...)        \
    [&] {                                     \
      if (HEADDIM == 32) {                    \
        constexpr static int kHeadDim = 32;   \
        return __VA_ARGS__();                 \
      } else if (HEADDIM == 64) {            \
        constexpr static int kHeadDim = 64;   \
        return __VA_ARGS__();                 \
      } else if (HEADDIM == 96) {            \
        constexpr static int kHeadDim = 96;   \
        return __VA_ARGS__();                 \
      } else if (HEADDIM == 128) {           \
        constexpr static int kHeadDim = 128;  \
        return __VA_ARGS__();                 \
      } else if (HEADDIM == 256) {           \
        constexpr static int kHeadDim = 256;  \
        return __VA_ARGS__();                 \
      } else {                                \
        throw std::runtime_error("Unsupported head dimension (expected 32, 64, 96, 128, or 256)"); \
      }                                       \
    }()
#endif
