//  Copyright (c) 2016 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/preprocessor/cat.hpp>
#include <pika/type_support/always_void.hpp>

#include <type_traits>

/// This macro creates a boolean unary meta-function such that for
/// any type X, has_name<X>::value == true if and only if X is a
/// class type and has a nested type member x::name. The generated
/// trait ends up in a namespace where the macro itself has been
/// placed.
#define PIKA_HAS_XXX_TRAIT_DEF(Name)                                            \
    template <typename T, typename Enable = void>                              \
    struct PIKA_PP_CAT(has_, Name)                                              \
      : std::false_type                                                        \
    {                                                                          \
    };                                                                         \
                                                                               \
    template <typename T>                                                      \
    struct PIKA_PP_CAT(has_,                                                    \
        Name)<T, typename pika::util::always_void<typename T::Name>::type>      \
      : std::true_type                                                         \
    {                                                                          \
    };                                                                         \
                                                                               \
    template <typename T>                                                      \
    using PIKA_PP_CAT(PIKA_PP_CAT(has_, Name), _t) =                             \
        typename PIKA_PP_CAT(has_, Name)<T>::type;                              \
                                                                               \
    template <typename T>                                                      \
    inline constexpr bool PIKA_PP_CAT(PIKA_PP_CAT(has_, Name), _v) =             \
        PIKA_PP_CAT(has_, Name)<T>::value;                                      \
    /**/
