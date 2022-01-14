//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config/defines.hpp>
#include <pika/local/config/attributes.hpp>
#include <pika/local/config/version.hpp>
#include <pika/preprocessor/cat.hpp>
#include <pika/preprocessor/expand.hpp>

///////////////////////////////////////////////////////////////////////////////
// Deprecate a given functionality starting pika V0.1
#if !defined(PIKA_HAVE_DEPRECATION_WARNINGS_V0_1)
#define PIKA_HAVE_DEPRECATION_WARNINGS_V0_1 1
#endif

#if (PIKA_VERSION_FULL >= 0x000100) &&                                    \
    (PIKA_HAVE_DEPRECATION_WARNINGS_V0_1 != 0)
#define PIKA_DEPRECATED_MSG_V0_1                                          \
    "This functionality is deprecated starting pika V0.1 and will be "     \
    "removed "                                                                 \
    "in the future. You can define "                                           \
    "PIKA_HAVE_DEPRECATION_WARNINGS_V0_1=0 to "                           \
    "acknowledge that you have received this warning."
#define PIKA_DEPRECATED_V0_1(x)                                           \
    PIKA_DEPRECATED(                                                      \
        x " (" PIKA_PP_EXPAND(PIKA_DEPRECATED_MSG_V0_1) ")")
#endif

#if !defined(PIKA_DEPRECATED_V0_1)
#define PIKA_DEPRECATED_V0_1(x)
#endif

///////////////////////////////////////////////////////////////////////////////
// Deprecate a given functionality starting at the given version of pika
#define PIKA_DEPRECATED_V(major, minor, x)                                \
    PIKA_PP_CAT(                                                                \
        PIKA_PP_CAT(PIKA_PP_CAT(PIKA_DEPRECATED_V, major), _), minor)       \
    (x)
