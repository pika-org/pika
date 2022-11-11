//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_DATAPAR)
#include <pika/type_support/pack.hpp>

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace pika::parallel::traits::detail {
    template <typename T, typename Enable = void>
    struct is_vector_pack : std::false_type
    {
    };

    template <typename T, typename Enable = void>
    struct is_scalar_vector_pack;

    template <typename T, typename Enable>
    struct is_scalar_vector_pack : std::false_type
    {
    };

    template <typename T, typename Enable = void>
    struct is_non_scalar_vector_pack : std::false_type
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct vector_pack_alignment;

    template <typename... Vector>
    struct vector_pack_alignment<std::tuple<Vector...>,
        std::enable_if_t<
            pika::util::detail::all_of_v<is_vector_pack<Vector>...>>>
    {
        using pack_type =
            typename std::tuple_element<0, std::tuple<Vector...>>::type;

        static std::size_t const value =
            vector_pack_alignment<pack_type>::value;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct vector_pack_size;

    template <typename... Vector>
    struct vector_pack_size<std::tuple<Vector...>,
        std::enable_if_t<
            pika::util::detail::all_of_v<is_vector_pack<Vector>...>>>
    {
        typedef typename std::tuple_element<0, std::tuple<Vector...>>::type
            pack_type;

        static std::size_t const value = vector_pack_size<pack_type>::value;
    };
}    // namespace pika::parallel::traits::detail

#if !defined(__CUDACC__)
#include <pika/parallel/util/detail/simd/vector_pack_alignment_size.hpp>
#endif

#endif
