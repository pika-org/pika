//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include <pika/preprocessor/cat.hpp>

#if defined(PIKA_HAVE_JEMALLOC_PREFIX)
// this is currently used only for jemalloc and if a special API prefix is
// used for its APIs
# include <jemalloc/jemalloc.h>
#endif

#include <pika/config/warnings_prefix.hpp>

namespace pika::detail {
#if defined(PIKA_HAVE_JEMALLOC_PREFIX)
    ///////////////////////////////////////////////////////////////////////////
    template <typename T = int>
    struct internal_allocator
    {
        using value_type = T;
        using pointer = T*;
        using const_pointer = T const*;
        using reference = T&;
        using const_reference = T const&;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        template <typename U>
        struct rebind
        {
            using other = internal_allocator<U>;
        };

        using is_always_equal = std::true_type;
        using propagate_on_container_move_assignment = std::true_type;

        internal_allocator() = default;

        template <typename U>
        explicit internal_allocator(internal_allocator<U> const&)
        {
        }

        pointer address(reference x) const noexcept { return &x; }

        const_pointer address(const_reference x) const noexcept { return &x; }

        [[nodiscard]] pointer allocate(size_type n, void const* hint = nullptr)
        {
            if (max_size() < n) { throw std::bad_array_new_length(); }

            pointer p = reinterpret_cast<pointer>(
                PIKA_PP_CAT(PIKA_HAVE_JEMALLOC_PREFIX, malloc)(n * sizeof(T)));
            if (p == nullptr) { throw std::bad_alloc(); }
            return p;
        }

        void deallocate(pointer p, size_type n) { PIKA_PP_CAT(PIKA_HAVE_JEMALLOC_PREFIX, free)(p); }

        size_type max_size() const noexcept
        {
            return (std::numeric_limits<size_type>::max)() / sizeof(T);
        }

        template <typename U, typename... Args>
        void construct(U* p, Args&&... args)
        {
            ::new ((void*) p) U(std::forward<Args>(args)...);
        }

        template <typename U>
        void destroy(U* p)
        {
            p->~U();
        }
    };

    template <typename T>
    constexpr bool operator==(internal_allocator<T> const&, internal_allocator<T> const&)
    {
        return true;
    }

    template <typename T>
    constexpr bool operator!=(internal_allocator<T> const&, internal_allocator<T> const&)
    {
        return false;
    }
#else
    // fall back to system allocator if no special internal allocator is needed
    template <typename T = int>
    using internal_allocator = std::allocator<T>;
#endif
}    // namespace pika::detail

#include <pika/config/warnings_suffix.hpp>
