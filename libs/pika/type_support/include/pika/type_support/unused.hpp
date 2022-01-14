/*=============================================================================
    Copyright (c) 2001-2011 Joel de Guzman
    Copyright (c) 2007-2021 Hartmut Kaiser

//  SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
==============================================================================*/
#pragma once

// clang-format off
#include <pika/local/config.hpp>
#if defined(PIKA_MSVC)
# pragma warning(push)
# pragma warning(disable: 4522) // multiple assignment operators specified warning
#endif
// clang-format on

namespace pika { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    // We do not import fusion::unused_type anymore to avoid boost::fusion
    // being turned into an associate namespace, as this interferes with ADL
    // in unexpected ways. We rather copy the full unused_type implementation.
    ///////////////////////////////////////////////////////////////////////////
    struct unused_type
    {
        constexpr PIKA_HOST_DEVICE PIKA_FORCEINLINE unused_type() noexcept =
            default;

        constexpr PIKA_HOST_DEVICE PIKA_FORCEINLINE unused_type(
            unused_type const&) noexcept
        {
        }
        constexpr PIKA_HOST_DEVICE PIKA_FORCEINLINE unused_type(
            unused_type&&) noexcept
        {
        }

        template <typename T>
        constexpr PIKA_HOST_DEVICE PIKA_FORCEINLINE unused_type(T const&) noexcept
        {
        }

        template <typename T>
        constexpr PIKA_HOST_DEVICE PIKA_FORCEINLINE unused_type const& operator=(
            T const&) const noexcept
        {
            return *this;
        }
        template <typename T>
        PIKA_HOST_DEVICE PIKA_FORCEINLINE unused_type& operator=(
            T const&) noexcept
        {
            return *this;
        }

        constexpr PIKA_HOST_DEVICE PIKA_FORCEINLINE unused_type const& operator=(
            unused_type const&) const noexcept
        {
            return *this;
        }
        constexpr PIKA_HOST_DEVICE PIKA_FORCEINLINE unused_type const& operator=(
            unused_type&&) const noexcept
        {
            return *this;
        }

        PIKA_HOST_DEVICE PIKA_FORCEINLINE unused_type& operator=(
            unused_type const&) noexcept
        {
            return *this;
        }
        PIKA_HOST_DEVICE PIKA_FORCEINLINE unused_type& operator=(
            unused_type&&) noexcept
        {
            return *this;
        }
    };

#if defined(PIKA_MSVC_NVCC)
    PIKA_CONSTANT
#endif
    constexpr unused_type unused = unused_type();
}}    // namespace pika::util

//////////////////////////////////////////////////////////////////////////////
// use this to silence compiler warnings related to unused function arguments.
#if defined(__CUDA_ARCH__)
#define PIKA_UNUSED(x) (void) x
#else
#define PIKA_UNUSED(x) ::pika::util::unused = (x)
#endif

/////////////////////////////////////////////////////////////
// use this to silence compiler warnings for global variables
#define PIKA_MAYBE_UNUSED [[maybe_unused]]

#if defined(PIKA_MSVC)
#pragma warning(pop)
#endif
