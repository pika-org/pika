//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Matthias Kretz
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_DATAPAR_VC)

#include <cstddef>
#include <type_traits>

#include <Vc/global.h>

#if defined(Vc_IS_VERSION_1) && Vc_IS_VERSION_1

#include <Vc/Vc>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace parallel { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename T, std::size_t N, typename Abi>
        struct vector_pack_type
        {
            using type = Vc::SimdArray<T, N>;
        };

        template <typename T, typename Abi>
        struct vector_pack_type<T, 0, Abi>
        {
            typedef typename std::conditional<std::is_void<Abi>::value,
                Vc::VectorAbi::Best<T>, Abi>::type abi_type;

            using type = Vc::Vector<T, abi_type>;
        };

        template <typename T, typename Abi>
        struct vector_pack_type<T, 1, Abi>
        {
            using type = Vc::Scalar::Vector<T>;
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, std::size_t N, typename Abi>
    struct vector_pack_type : detail::vector_pack_type<T, N, Abi>
    {
    };

    // don't wrap types twice
    template <typename T, std::size_t N, typename Abi1, typename Abi2>
    struct vector_pack_type<Vc::Vector<T, Abi1>, N, Abi2>
    {
        using type = Vc::Vector<T, Abi1>;
    };

    template <typename T, std::size_t N1, typename V, std::size_t W,
        std::size_t N2, typename Abi>
    struct vector_pack_type<Vc::SimdArray<T, N1, V, W>, N2, Abi>
    {
        typedef Vc::SimdArray<T, N1, V, W> type;
    };

    template <typename T, std::size_t N, typename Abi>
    struct vector_pack_type<Vc::Scalar::Vector<T>, N, Abi>
    {
        using type = Vc::Scalar::Vector<T>;
    };
}}}    // namespace pika::parallel::traits

#else

#include <Vc/datapar>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace parallel { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        // specifying both, N and an Abi is not allowed
        template <typename T, std::size_t N, typename Abi>
        struct vector_pack_type;

        template <typename T, std::size_t N>
        struct vector_pack_type<T, N, void>
        {
            using type = Vc::datapar<T, Vc::datapar_abi::fixed_size<N>>;
        };

        template <typename T, typename Abi>
        struct vector_pack_type<T, 0, Abi>
        {
            typedef typename std::conditional<std::is_void<Abi>::value,
                Vc::datapar_abi::native<T>, Abi>::type abi_type;

            using type = Vc::datapar<T, abi_type>;
        };

        template <typename T, typename Abi>
        struct vector_pack_type<T, 1, Abi>
        {
            using type = Vc::datapar<T, Vc::datapar_abi::scalar>;
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, std::size_t N, typename Abi>
    struct vector_pack_type : detail::vector_pack_type<T, N, Abi>
    {
    };

    // don't wrap types twice
    template <typename T, std::size_t N, typename Abi1, typename Abi2>
    struct vector_pack_type<Vc::datapar<T, Abi1>, N, Abi2>
    {
        using type = Vc::datapar<T, Abi1>;
    };
}}}    // namespace pika::parallel::traits

#endif    // Vc_IS_VERSION_1

#endif
