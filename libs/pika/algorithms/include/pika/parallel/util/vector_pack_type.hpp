//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_DATAPAR)

#include <cstddef>
#include <tuple>

namespace pika::parallel::traits::detail {
    template <typename T, std::size_t N = 0, typename Abi = void>
    struct vector_pack_type;

    // handle tuple<> transformations
    template <typename... T, std::size_t N, typename Abi>
    struct vector_pack_type<std::tuple<T...>, N, Abi>
    {
        using type = std::tuple<typename vector_pack_type<T, N, Abi>::type...>;
    };

    template <typename T, typename NewT>
    struct rebind_pack
    {
        using type = typename vector_pack_type<T>::type;
    };
}    // namespace pika::parallel::traits::detail

#if !defined(__CUDACC__)
#include <pika/parallel/util/detail/simd/vector_pack_type.hpp>
#endif

#endif
