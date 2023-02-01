//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config/defines.hpp>
#include <pika/config/compiler_specific.hpp>

/// This macro evaluates to ``inline constexpr`` for host code and
/// ``__device__ static const`` for device code with NVCC
#if defined(PIKA_COMPUTE_DEVICE_CODE) && defined(__NVCC__)
#define PIKA_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE PIKA_DEVICE static const
#else
#define PIKA_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE inline constexpr
#endif
