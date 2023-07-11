//  Copyright (c) 2020 Thomas Heller
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

// NOLINTNEXTLINE(bugprone-reserved-identifier)
inline void* __aligned_alloc(std::size_t alignment, std::size_t size) noexcept
{
    return PIKA_PP_CAT(PIKA_HAVE_JEMALLOC_PREFIX, aligned_alloc)(alignment, size);
}

// NOLINTNEXTLINE(bugprone-reserved-identifier)
inline void __aligned_free(void* p) noexcept
{
    return PIKA_PP_CAT(PIKA_HAVE_JEMALLOC_PREFIX, free)(p);
}

#elif defined(PIKA_HAVE_CXX17_STD_ALIGNED_ALLOC)

# include <cstdlib>

// NOLINTNEXTLINE(bugprone-reserved-identifier)
inline void* __aligned_alloc(std::size_t alignment, std::size_t size) noexcept
{
    return std::aligned_alloc(alignment, size);
}

// NOLINTNEXTLINE(bugprone-reserved-identifier)
inline void __aligned_free(void* p) noexcept { std::free(p); }

#elif defined(PIKA_HAVE_C11_ALIGNED_ALLOC)

# include <stdlib.h>

// NOLINTNEXTLINE(bugprone-reserved-identifier)
inline void* __aligned_alloc(std::size_t alignment, std::size_t size) noexcept
{
    return aligned_alloc(alignment, size);
}

// NOLINTNEXTLINE(bugprone-reserved-identifier)
inline void __aligned_free(void* p) noexcept { free(p); }

#else    // !PIKA_HAVE_CXX17_STD_ALIGNED_ALLOC && !PIKA_HAVE_C11_ALIGNED_ALLOC

# include <cstdlib>

// provide our own (simple) implementation of aligned_alloc
// NOLINTNEXTLINE(bugprone-reserved-identifier)
inline void* __aligned_alloc(std::size_t alignment, std::size_t size) noexcept
{
    if (alignment < alignof(void*)) { alignment = alignof(void*); }

    std::size_t space = size + alignment - 1;
    void* allocated_mem = std::malloc(space + sizeof(void*));
    if (allocated_mem == nullptr) { return nullptr; }

    void* aligned_mem = static_cast<void*>(static_cast<char*>(allocated_mem) + sizeof(void*));

    std::align(alignment, size, aligned_mem, space);
    *(static_cast<void**>(aligned_mem) - 1) = allocated_mem;

    return aligned_mem;
}

// NOLINTNEXTLINE(bugprone-reserved-identifier)
inline void __aligned_free(void* p) noexcept
{
    if (nullptr != p) { std::free(*(static_cast<void**>(p) - 1)); }
}

#endif

#include <pika/config/warnings_prefix.hpp>

namespace pika::detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T = int>
    struct aligned_allocator
    {
        using value_type = T;
        using pointer = T*;
        using const_pointer = const T*;
        using reference = T&;
        using const_reference = T const&;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        template <typename U>
        struct rebind
        {
            using other = aligned_allocator<U>;
        };

        using is_always_equal = std::true_type;
        using propagate_on_container_move_assignment = std::true_type;

        aligned_allocator() = default;

        template <typename U>
        explicit aligned_allocator(aligned_allocator<U> const&)
        {
        }

        pointer address(reference x) const noexcept { return &x; }

        const_pointer address(const_reference x) const noexcept { return &x; }

        [[nodiscard]] pointer allocate(size_type n, void const* = nullptr)
        {
            if (max_size() < n) { throw std::bad_array_new_length(); }

            pointer p = reinterpret_cast<pointer>(__aligned_alloc(alignof(T), n * sizeof(T)));

            if (p == nullptr) { throw std::bad_alloc(); }

            return p;
        }

        void deallocate(pointer p, size_type) { __aligned_free(p); }

        size_type max_size() const noexcept
        {
            return (std::numeric_limits<size_type>::max)() / sizeof(T);
        }

        template <typename U, typename... Args>
        void construct(U* p, Args&&... args)
        {
            ::new ((void*) p) U(PIKA_FORWARD(Args, args)...);
        }

        template <typename U>
        void destroy(U* p)
        {
            p->~U();
        }
    };

    template <typename T>
    constexpr bool operator==(aligned_allocator<T> const&, aligned_allocator<T> const&)
    {
        return true;
    }

    template <typename T>
    constexpr bool operator!=(aligned_allocator<T> const&, aligned_allocator<T> const&)
    {
        return false;
    }
}    // namespace pika::detail

#include <pika/config/warnings_suffix.hpp>
