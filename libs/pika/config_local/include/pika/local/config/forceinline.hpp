//  Copyright (c) 2012-2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config/compiler_specific.hpp>

#if defined(DOXYGEN)
/// Marks a function to be forced inline.
#define PIKA_FORCEINLINE
#else

// clang-format off
#if !defined(PIKA_FORCEINLINE)
#   if defined(__NVCC__) || defined(__CUDACC__)
#       define PIKA_FORCEINLINE inline
#   elif defined(PIKA_MSVC)
#       define PIKA_FORCEINLINE __forceinline
#   elif defined(__GNUC__)
#       define PIKA_FORCEINLINE inline __attribute__ ((__always_inline__))
#   else
#       define PIKA_FORCEINLINE inline
#   endif
#endif
// clang-format on
#endif
