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

#include <pika/local/config.hpp>
#include <pika/concurrency/cache_line_data.hpp>
#include <pika/hashing/fibhash.hpp>
#include <pika/lock_registration/detail/register_locks.hpp>
#include <pika/modules/itt_notify.hpp>
#include <pika/thread_support/spinlock.hpp>

#include <cstddef>

namespace pika { namespace util {

    namespace detail {
#if PIKA_HAVE_ITTNOTIFY != 0
        template <typename Tag, std::size_t N>
        struct itt_spinlock_init
        {
            itt_spinlock_init();
            ~itt_spinlock_init();
        };
#endif
    }    // namespace detail

    template <typename Tag, std::size_t N = PIKA_HAVE_SPINLOCK_POOL_NUM>
    class spinlock_pool
    {
    private:
        static cache_aligned_data<detail::spinlock> pool_[N];
#if PIKA_HAVE_ITTNOTIFY != 0
        static detail::itt_spinlock_init<Tag, N> init_;
#endif

    public:
        static detail::spinlock& spinlock_for(void const* pv)
        {
            std::size_t i = fibhash<N>(reinterpret_cast<std::size_t>(pv));
            return pool_[i].data_;
        }
    };

    template <typename Tag, std::size_t N>
    cache_aligned_data<detail::spinlock> spinlock_pool<Tag, N>::pool_[N];

#if PIKA_HAVE_ITTNOTIFY != 0
    namespace detail {
        template <typename Tag, std::size_t N>
        itt_spinlock_init<Tag, N>::itt_spinlock_init()
        {
            for (int i = 0; i < N; ++i)
            {
                PIKA_ITT_SYNC_CREATE((&spinlock_pool<Tag, N>::pool_[i].data_),
                    "util::detail::spinlock", 0);
                PIKA_ITT_SYNC_RENAME((&spinlock_pool<Tag, N>::pool_[i].data_),
                    "util::detail::spinlock");
            }
        }

        template <typename Tag, std::size_t N>
        itt_spinlock_init<Tag, N>::~itt_spinlock_init()
        {
            for (int i = 0; i < N; ++i)
            {
                PIKA_ITT_SYNC_DESTROY((&spinlock_pool<Tag, N>::pool_[i].data_));
            }
        }
    }    // namespace detail

    template <typename Tag, std::size_t N>
    util::detail::itt_spinlock_init<Tag, N> spinlock_pool<Tag, N>::init_;
#endif

}}    // namespace pika::util
