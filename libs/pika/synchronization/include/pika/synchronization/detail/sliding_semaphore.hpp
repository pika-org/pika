//  Copyright (c) 2016-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/synchronization/detail/condition_variable.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/concurrency/spinlock.hpp>

#include <cstdint>
#include <mutex>
#include <utility>
#endif

#if defined(PIKA_MSVC_WARNING_PRAGMA)
# pragma warning(push)
# pragma warning(disable : 4251)
#endif

////////////////////////////////////////////////////////////////////////////////
namespace pika::detail {

    class sliding_semaphore
    {
    private:
        using mutex_type = pika::concurrency::detail::spinlock;

    public:
        PIKA_EXPORT sliding_semaphore(std::int64_t max_difference, std::int64_t lower_limit);
        PIKA_EXPORT ~sliding_semaphore();

        PIKA_EXPORT void set_max_difference(
            std::unique_lock<mutex_type>& l, std::int64_t max_difference, std::int64_t lower_limit);

        PIKA_EXPORT void wait(std::unique_lock<mutex_type>& l, std::int64_t upper_limit);

        PIKA_EXPORT bool try_wait(std::unique_lock<mutex_type>& l, std::int64_t upper_limit);

        PIKA_EXPORT void signal(std::unique_lock<mutex_type> l, std::int64_t lower_limit);

        PIKA_EXPORT std::int64_t signal_all(std::unique_lock<mutex_type> l);

    private:
        std::int64_t max_difference_;
        std::int64_t lower_limit_;
        pika::detail::condition_variable cond_;
    };
}    // namespace pika::detail

#if defined(PIKA_MSVC_WARNING_PRAGMA)
# pragma warning(pop)
#endif
