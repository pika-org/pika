//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

#if defined(PIKA_HAVE_DATAPAR)
#include <pika/datastructures/tuple.hpp>
#include <pika/type_support/pack.hpp>

#include <cstddef>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace parallel { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
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
    struct vector_pack_alignment<pika::tuple<Vector...>,
        typename std::enable_if<
            pika::util::all_of<is_vector_pack<Vector>...>::value>::type>
    {
        typedef typename pika::tuple_element<0, pika::tuple<Vector...>>::type
            pack_type;

        static std::size_t const value =
            vector_pack_alignment<pack_type>::value;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct vector_pack_size;

    template <typename... Vector>
    struct vector_pack_size<pika::tuple<Vector...>,
        typename std::enable_if<
            pika::util::all_of<is_vector_pack<Vector>...>::value>::type>
    {
        typedef typename pika::tuple_element<0, pika::tuple<Vector...>>::type
            pack_type;

        static std::size_t const value = vector_pack_size<pack_type>::value;
    };
}}}    // namespace pika::parallel::traits

#if !defined(__CUDACC__)
#include <pika/execution/traits/detail/simd/vector_pack_alignment_size.hpp>
#include <pika/execution/traits/detail/vc/vector_pack_alignment_size.hpp>
#endif

#endif
