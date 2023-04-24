//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/concurrency/spinlock.hpp>
#include <pika/synchronization/detail/condition_variable.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>

#if defined(PIKA_MSVC_WARNING_PRAGMA)
# pragma warning(push)
# pragma warning(disable : 4251)
#endif

////////////////////////////////////////////////////////////////////////////////
namespace pika::detail {

    class counting_semaphore
    {
    private:
        using mutex_type = pika::detail::spinlock;

    public:
        PIKA_EXPORT counting_semaphore(std::ptrdiff_t value = 0);
        PIKA_EXPORT ~counting_semaphore();

        PIKA_EXPORT void wait(std::unique_lock<mutex_type>& l, std::ptrdiff_t count);

        PIKA_EXPORT bool wait_until(std::unique_lock<mutex_type>& l,
            pika::chrono::steady_time_point const& abs_time, std::ptrdiff_t count);

        PIKA_EXPORT bool try_wait(std::unique_lock<mutex_type>& l, std::ptrdiff_t count = 1);

        PIKA_EXPORT bool try_acquire(std::unique_lock<mutex_type>& l);

        PIKA_EXPORT void signal(std::unique_lock<mutex_type> l, std::ptrdiff_t count);

        PIKA_EXPORT std::ptrdiff_t signal_all(std::unique_lock<mutex_type> l);

    private:
        std::ptrdiff_t value_;
        pika::detail::condition_variable cond_;
    };
}    // namespace pika::detail

#if defined(PIKA_MSVC_WARNING_PRAGMA)
# pragma warning(pop)
#endif
