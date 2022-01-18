#pragma once

//
//  Copyright 2002-2018 Peter Dimov
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt
//
//  http://www.boost.org/libs/assert
//
//  Adapted to pika naming scheme

#include <pika/local/config.hpp>

namespace pika { namespace assertion { namespace detail {

    constexpr inline void current_function_helper()
    {
#if defined(__GNUC__) || (defined(__MWERKS__) && (__MWERKS__ >= 0x3000)) ||    \
    (defined(__ICC) && (__ICC >= 600)) || defined(__ghs__) ||                  \
    defined(__clang__)

#define PIKA_ASSERT_CURRENT_FUNCTION __PRETTY_FUNCTION__

#elif defined(__DMC__) && (__DMC__ >= 0x810)

#define PIKA_ASSERT_CURRENT_FUNCTION __PRETTY_FUNCTION__

#elif defined(__FUNCSIG__)

#define PIKA_ASSERT_CURRENT_FUNCTION __FUNCSIG__

#elif (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600)) ||              \
    (defined(__IBMCPP__) && (__IBMCPP__ >= 500))

#define PIKA_ASSERT_CURRENT_FUNCTION __FUNCTION__

#elif defined(__BORLANDC__) && (__BORLANDC__ >= 0x550)

#define PIKA_ASSERT_CURRENT_FUNCTION __FUNC__

#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)

#define PIKA_ASSERT_CURRENT_FUNCTION __func__

#elif defined(__cplusplus) && (__cplusplus >= 201103)

#define PIKA_ASSERT_CURRENT_FUNCTION __func__

#else

#define PIKA_ASSERT_CURRENT_FUNCTION "(unknown)"

#endif
    }

}}}    // namespace pika::assertion::detail
