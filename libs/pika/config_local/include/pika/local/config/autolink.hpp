//  Copyright (c) 2005-2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config/compiler_specific.hpp>
#include <pika/local/config/debug.hpp>

// enable auto-linking for supported platforms
#if defined(PIKA_MSVC) || defined(__BORLANDC__) ||                              \
    (defined(__MWERKS__) && defined(_WIN32) && (__MWERKS__ >= 0x3000)) ||      \
    (defined(__ICL) && defined(_MSC_EXTENSIONS) && (PIKA_MSVC >= 1200))

#if !defined(PIKA_AUTOLINK_LIB_NAME)
#error "Macro PIKA_AUTOLINK_LIB_NAME not set (internal error)"
#endif

#if defined(PIKA_DEBUG)
#pragma comment(lib,                                                           \
    PIKA_AUTOLINK_LIB_NAME "d"                                                  \
                          ".lib")
#else
#pragma comment(lib, PIKA_AUTOLINK_LIB_NAME ".lib")
#endif

#endif

#undef PIKA_AUTOLINK_LIB_NAME
