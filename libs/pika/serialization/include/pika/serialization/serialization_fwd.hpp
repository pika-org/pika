//  Copyright (c) 2015 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/preprocessor/strip_parens.hpp>
#include <pika/serialization/config/defines.hpp>

#include <type_traits>

namespace pika { namespace serialization {

    class access;

    struct input_archive;
    struct output_archive;

    struct binary_filter;

    template <typename T>
    output_archive& operator<<(output_archive& ar, T const& t);

    template <typename T>
    input_archive& operator>>(input_archive& ar, T& t);

    template <typename T>
    output_archive& operator&(output_archive& ar, T const& t);

    template <typename T>
    input_archive& operator&(input_archive& ar, T& t);

}}    // namespace pika::serialization

#define PIKA_SERIALIZATION_SPLIT_MEMBER()                                       \
    void serialize(pika::serialization::input_archive& ar, unsigned)            \
    {                                                                          \
        load(ar, 0);                                                           \
    }                                                                          \
    void serialize(pika::serialization::output_archive& ar, unsigned) const     \
    {                                                                          \
        save(ar, 0);                                                           \
    }                                                                          \
    /**/

#define PIKA_SERIALIZATION_SPLIT_FREE(T)                                        \
    PIKA_FORCEINLINE void serialize(                                            \
        pika::serialization::input_archive& ar, T& t, unsigned)                 \
    {                                                                          \
        load(ar, t, 0);                                                        \
    }                                                                          \
    PIKA_FORCEINLINE void serialize(                                            \
        pika::serialization::output_archive& ar, T& t, unsigned)                \
    {                                                                          \
        save(ar, const_cast<std::add_const_t<T>&>(t), 0);                      \
    }                                                                          \
    /**/

#define PIKA_SERIALIZATION_SPLIT_FREE_TEMPLATE(TEMPLATE, ARGS)                  \
    PIKA_PP_STRIP_PARENS(TEMPLATE)                                              \
    PIKA_FORCEINLINE void serialize(pika::serialization::input_archive& ar,      \
        PIKA_PP_STRIP_PARENS(ARGS) & t, unsigned)                               \
    {                                                                          \
        load(ar, t, 0);                                                        \
    }                                                                          \
    PIKA_PP_STRIP_PARENS(TEMPLATE)                                              \
    PIKA_FORCEINLINE void serialize(pika::serialization::output_archive& ar,     \
        PIKA_PP_STRIP_PARENS(ARGS) & t, unsigned)                               \
    {                                                                          \
        save(ar, const_cast<std::add_const_t<PIKA_PP_STRIP_PARENS(ARGS)>&>(t),  \
            0);                                                                \
    }                                                                          \
    /**/
