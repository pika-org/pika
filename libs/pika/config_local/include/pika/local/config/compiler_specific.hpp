//  Copyright (c) 2012 Maciej Brodowicz
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config/defines.hpp>

#if defined(DOXYGEN)
/// Returns the GCC version pika is compiled with. Only set if compiled with GCC.
#define PIKA_GCC_VERSION
/// Returns the Clang version pika is compiled with. Only set if compiled with
/// Clang.
#define PIKA_CLANG_VERSION
/// Returns the Intel Compiler version pika is compiled with. Only set if
/// compiled with the Intel Compiler.
#define PIKA_INTEL_VERSION
/// This macro is set if the compilation is with MSVC.
#define PIKA_MSVC
/// This macro is set if the compilation is with Mingw.
#define PIKA_MINGW
/// This macro is set if the compilation is for Windows.
#define PIKA_WINDOWS
/// This macro is set if the compilation is for Intel Knights Landing.
#define PIKA_NATIVE_MIC
#else

// clang-format off
#if defined(__GNUC__)

// macros to facilitate handling of compiler-specific issues
#  define PIKA_GCC_VERSION (__GNUC__*10000 + __GNUC_MINOR__*100 + __GNUC_PATCHLEVEL__)

#  define PIKA_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS 1

#  undef PIKA_CLANG_VERSION
#  undef PIKA_INTEL_VERSION

#else

#  undef PIKA_GCC_VERSION

#endif

#if defined(__clang__)

#  define PIKA_CLANG_VERSION \
 (__clang_major__*10000 + __clang_minor__*100 + __clang_patchlevel__)

#  undef PIKA_INTEL_VERSION

#else

#  undef PIKA_CLANG_VERSION

#endif

#if defined(__INTEL_COMPILER)
# define PIKA_INTEL_VERSION __INTEL_COMPILER
# if defined(_WIN32) || (_WIN64)
#  define PIKA_INTEL_WIN PIKA_INTEL_VERSION
// suppress a couple of benign warnings
   // template parameter "..." is not used in declaring the parameter types of
   // function template "..."
#  pragma warning disable 488
   // invalid redeclaration of nested class
#  pragma warning disable 1170
   // decorated name length exceeded, name was truncated
#  pragma warning disable 2586
# endif
#else

#  undef PIKA_INTEL_VERSION

#endif

// Identify if we compile for the MIC
#if defined(__MIC)
#   define PIKA_NATIVE_MIC
#endif

#if defined(_MSC_VER)
#  define PIKA_WINDOWS
#  define PIKA_MSVC _MSC_VER
#  define PIKA_MSVC_WARNING_PRAGMA
#  if defined(__NVCC__)
#    define PIKA_MSVC_NVCC
#  endif
#  define PIKA_CDECL __cdecl
#endif

#if defined(__MINGW32__)
#   define PIKA_WINDOWS
#   define PIKA_MINGW
#endif

// Detecting CUDA compilation mode
// Detecting NVCC
#if defined(__NVCC__) || defined(__CUDACC__)
// NVCC build version numbers can be high (without limit?) so we leave it out
// from the version definition
#  define PIKA_CUDA_VERSION (__CUDACC_VER_MAJOR__*100 + __CUDACC_VER_MINOR__)
#  define PIKA_COMPUTE_CODE
#  if defined(__CUDA_ARCH__)
     // nvcc compiling CUDA code, device mode.
#    define PIKA_COMPUTE_DEVICE_CODE
#  endif
// Detecting Clang CUDA
#elif defined(__clang__) && defined(__CUDA__)
#  define PIKA_COMPUTE_CODE
#  if defined(__CUDA_ARCH__)
     // clang compiling CUDA code, device mode.
#    define PIKA_COMPUTE_DEVICE_CODE
#  endif
// Detecting HIPCC
#elif defined(__HIPCC__)
#  include <hip/hip_version.h>
#  define PIKA_HIP_VERSION HIP_VERSION
#  if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdeprecated-copy"
#    pragma clang diagnostic ignored "-Wunused-parameter"
#  endif
   // Not like nvcc, the __device__ __host__ function decorators are not defined
   // by the compiler
#  include <hip/hip_runtime_api.h>
#  if defined(__clang__)
#    pragma clang diagnostic pop
#  endif
#  define PIKA_COMPUTE_CODE
#  if defined(__HIP_DEVICE_COMPILE__)
     // hipclang compiling CUDA/HIP code, device mode.
#    define PIKA_COMPUTE_DEVICE_CODE
#  endif
#endif

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
#  define PIKA_COMPUTE_HOST_CODE
#endif

#if defined(PIKA_COMPUTE_CODE)
#define PIKA_DEVICE __device__
#define PIKA_HOST __host__
#define PIKA_CONSTANT __constant__
#else
#define PIKA_DEVICE
#define PIKA_HOST
#define PIKA_CONSTANT
#endif
#define PIKA_HOST_DEVICE PIKA_HOST PIKA_DEVICE


#if !defined(PIKA_CDECL)
#define PIKA_CDECL
#endif

#if defined(PIKA_HAVE_SANITIZERS)
#  if defined(__has_feature)
#  if __has_feature(address_sanitizer)
#      define PIKA_HAVE_ADDRESS_SANITIZER
#    endif
#  elif defined(__SANITIZE_ADDRESS__)   // MSVC defines this
#    define PIKA_HAVE_ADDRESS_SANITIZER
#  endif
#endif
// clang-format on
#endif
