//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if !defined(PIKA_HAVE_MODULE)
#include <boost/version.hpp>
#include <boost/lockfree/policies.hpp>
#include <boost/lockfree/queue.hpp>

#include <cstddef>
#endif

namespace pika::concurrency::detail {
    template <typename T, typename Alloc = std::allocator<T>>
    class caching_freelist : public boost::lockfree::detail::freelist_stack<T, Alloc>
    {
        using base_type = boost::lockfree::detail::freelist_stack<T, Alloc>;

    public:
        caching_freelist(std::size_t n = 0)
          : boost::lockfree::detail::freelist_stack<T, Alloc>(Alloc(), n)
        {
        }

        T* allocate() { return this->base_type::template allocate<true, false>(); }

        void deallocate(T* n) { this->base_type::template deallocate<true>(n); }
    };

    template <typename T, typename Alloc = std::allocator<T>>
    class static_freelist : public boost::lockfree::detail::freelist_stack<T, Alloc>
    {
        using base_type = boost::lockfree::detail::freelist_stack<T, Alloc>;

    public:
        static_freelist(std::size_t n = 0)
          : boost::lockfree::detail::freelist_stack<T, Alloc>(Alloc(), n)
        {
        }

        T* allocate() { return this->base_type::template allocate<true, true>(); }

        void deallocate(T* n) { this->base_type::template deallocate<true>(n); }
    };

    struct caching_freelist_t
    {
    };
    struct static_freelist_t
    {
    };
}    // namespace pika::concurrency::detail
