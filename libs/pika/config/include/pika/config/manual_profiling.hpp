////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

// clang-format off
#if defined(__GNUC__)
  #define PIKA_SUPER_PURE  __attribute__((const))
  #define PIKA_PURE        __attribute__((pure))
  #define PIKA_HOT         __attribute__((hot))
  #define PIKA_COLD        __attribute__((cold))
#else
  #define PIKA_SUPER_PURE
  #define PIKA_PURE
  #define PIKA_HOT
  #define PIKA_COLD
#endif
// clang-format on
