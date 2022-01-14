//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

#if defined(PIKA_HAVE_DATAPAR)
#include <pika/datastructures/tuple.hpp>

#include <cstddef>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace parallel { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    // exposition only
    template <typename T, std::size_t N = 0, typename Abi = void>
    struct vector_pack_type;

    // handle tuple<> transformations
    template <typename... T, std::size_t N, typename Abi>
    struct vector_pack_type<pika::tuple<T...>, N, Abi>
    {
        typedef pika::tuple<typename vector_pack_type<T, N, Abi>::type...> type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename NewT>
    struct rebind_pack
    {
        typedef typename vector_pack_type<T>::type type;
    };
}}}    // namespace pika::parallel::traits

#if !defined(__CUDACC__)
#include <pika/execution/traits/detail/simd/vector_pack_type.hpp>
#include <pika/execution/traits/detail/vc/vector_pack_type.hpp>
#endif

#endif
