//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Parts of this have been taken from Eric Niebler's Ranges V3 library, see
// its project home at: https://github.com/ericniebler/range-v3.

#pragma once

#include <pika/preprocessor/cat.hpp>

#include <type_traits>

#define PIKA_CONCEPT_REQUIRES_(...)                                             \
    int PIKA_PP_CAT(_concept_requires_, __LINE__) = 42,                         \
                                       typename std::enable_if <               \
            (PIKA_PP_CAT(_concept_requires_, __LINE__) == 43) ||                \
        (__VA_ARGS__),                                                         \
                                       int >                                   \
        ::type PIKA_PP_CAT(_concept_check_, __LINE__) = 0 /**/

#define PIKA_CONCEPT_REQUIRES(...)                                              \
    template <int PIKA_PP_CAT(_concept_requires_, __LINE__) = 42,               \
        typename std::enable_if<                                               \
            (PIKA_PP_CAT(_concept_requires_, __LINE__) == 43) || (__VA_ARGS__), \
            int>::type PIKA_PP_CAT(_concept_check_, __LINE__) = 0> /**/

#define PIKA_CONCEPT_ASSERT(...)                                                \
    static_assert((__VA_ARGS__), "Concept check failed") /**/
