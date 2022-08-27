//  Copyright (c) 2021 Srinivas Yadav
//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_CXX20_EXPERIMENTAL_SIMD)

#include <experimental/simd>

#include <cstddef>
#include <type_traits>

namespace pika::parallel::traits::detail {
    template <typename T, std::size_t N, typename Abi>
    struct vector_pack_type_impl
    {
        using type = std::experimental::fixed_size_simd<T, N>;
    };

    template <typename T, typename Abi>
    struct vector_pack_type_impl<T, 0, Abi>
    {
        typedef typename std::conditional<std::is_void<Abi>::value,
            std::experimental::simd_abi::native<T>, Abi>::type abi_type;

        using type = std::experimental::simd<T, abi_type>;
    };

    template <typename T, typename Abi>
    struct vector_pack_type_impl<T, 1, Abi>
    {
        typedef std::experimental::simd<T, std::experimental::simd_abi::scalar>
            type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, std::size_t N, typename Abi>
    struct vector_pack_type : vector_pack_type_impl<T, N, Abi>
    {
    };
}    // namespace pika::parallel::traits::detail

#endif
