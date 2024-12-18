//  Copyright (c) 2012 Hartmut Kaiser
//
//  taken from:
//  boost/detail/spinlock_pool.hpp
//
//  Copyright (c) 2008 Peter Dimov
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/concurrency/cache_line_data.hpp>
#include <pika/hashing/fibhash.hpp>
#include <pika/lock_registration/detail/register_locks.hpp>
#include <pika/thread_support/spinlock.hpp>

#include <cstddef>

namespace pika::concurrency::detail {
    template <typename Tag, std::size_t N = PIKA_HAVE_SPINLOCK_POOL_NUM>
    class spinlock_pool
    {
    private:
        static pika::concurrency::detail::cache_aligned_data<::pika::detail::spinlock> pool_[N];

    public:
        static ::pika::detail::spinlock& spinlock_for(void const* pv)
        {
            std::size_t i = pika::detail::fibhash<N>(reinterpret_cast<std::size_t>(pv));
            return pool_[i].data_;
        }
    };

    template <typename Tag, std::size_t N>
    pika::concurrency::detail::cache_aligned_data<::pika::detail::spinlock>
        spinlock_pool<Tag, N>::pool_[N];
}    // namespace pika::concurrency::detail
