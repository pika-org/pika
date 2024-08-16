//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/execution_base/agent_base.hpp>
#include <pika/execution_base/agent_ref.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/timing/high_resolution_timer.hpp>
#include <pika/timing/steady_clock.hpp>
#endif

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ratio>

namespace pika::execution {
    namespace detail {
        PIKA_EXPORT agent_base& get_default_agent();
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace this_thread::detail {
        struct agent_storage;

        PIKA_EXPORT agent_storage* get_agent_storage();

        struct PIKA_EXPORT reset_agent
        {
            reset_agent(detail::agent_storage*, execution::detail::agent_base& impl);
            reset_agent(execution::detail::agent_base& impl);
            ~reset_agent();

            detail::agent_storage* storage_;
            execution::detail::agent_base* old_;
        };

        PIKA_EXPORT pika::execution::detail::agent_ref agent();

        PIKA_EXPORT void yield(char const* desc = "pika::execution::this_thread::detail::yield");
        PIKA_EXPORT void yield_k(
            std::size_t k, char const* desc = "pika::execution::this_thread::detail::yield_k");
        PIKA_EXPORT void spin_k(
            std::size_t k, char const* desc = "pika::execution::this_thread::detail::spin_k");
        PIKA_EXPORT void suspend(
            char const* desc = "pika::execution::this_thread::detail::suspend");

        template <typename Rep, typename Period>
        void sleep_for(std::chrono::duration<Rep, Period> const& sleep_duration,
            char const* desc = "pika::execution::this_thread::detail::sleep_for")
        {
            agent().sleep_for(sleep_duration, desc);
        }

        template <class Clock, class Duration>
        void sleep_until(std::chrono::time_point<Clock, Duration> const& sleep_time,
            char const* desc = "pika::execution::this_thread::detail::sleep_for")
        {
            agent().sleep_until(sleep_time, desc);
        }
    }    // namespace this_thread::detail
}    // namespace pika::execution

namespace pika::util {
    template <typename Predicate>
    void yield_while(Predicate&& predicate, const char* thread_name = nullptr,
        bool allow_timed_suspension = true)
    {
        auto yield_or_spin = allow_timed_suspension ?
            &pika::execution::this_thread::detail::yield_k :
            &pika::execution::this_thread::detail::spin_k;

        for (std::size_t k = 0; predicate(); ++k) { yield_or_spin(k, thread_name); }
    }

    namespace detail {
        // yield_while_count yields until the predicate returns true
        // required_count times consecutively. This function is used in cases
        // where there is a small false positive rate and repeatedly calling the
        // predicate reduces the rate of false positives overall.
        //
        // Note: This is mostly a hack used to work around the raciness of
        // termination detection for thread pools and the runtime and can be
        // replaced if and when a better solution appears.
        template <typename Predicate>
        void yield_while_count(Predicate&& predicate, std::size_t required_count,
            const char* thread_name = nullptr, bool allow_timed_suspension = true)
        {
            auto yield_or_spin = allow_timed_suspension ?
                &pika::execution::this_thread::detail::yield_k :
                &pika::execution::this_thread::detail::spin_k;

            std::size_t count = 0;
            for (std::size_t k = 0;; ++k)
            {
                if (!predicate())
                {
                    if (++count > required_count) { return; }
                }
                else
                {
                    count = 0;
                    yield_or_spin(k, thread_name);
                }
            }
        }

        // yield_while_count_timeout is similar to yield_while_count, with the
        // addition of a timeout parameter. If the timeout is exceeded, waiting
        // is stopped and the function returns false. If the predicate is
        // successfully waited for the function returns true.
        template <typename Predicate>
        [[nodiscard]] bool yield_while_count_timeout(Predicate&& predicate,
            std::size_t required_count, std::chrono::duration<double> timeout,
            const char* thread_name = nullptr, bool allow_timed_suspension = true)
        {
            // Seconds represented using a double
            using duration_type = std::chrono::duration<double>;

            const bool use_timeout = timeout >= duration_type(0.0);
            auto yield_or_spin = allow_timed_suspension ?
                &pika::execution::this_thread::detail::yield_k :
                &pika::execution::this_thread::detail::spin_k;

            std::size_t count = 0;
            pika::chrono::detail::high_resolution_timer t;

            for (std::size_t k = 0;; ++k)
            {
                if (use_timeout && duration_type(t.elapsed()) > timeout) { return false; }

                if (!predicate())
                {
                    if (++count > required_count) { return true; }
                }
                else
                {
                    count = 0;
                    yield_or_spin(k, thread_name);
                }
            }
        }

        // yield_while_timeout is similar to yield_while, with the
        // addition of a timeout parameter. If the timeout is exceeded, waiting
        // is stopped and the function returns false. If the predicate is
        // successfully waited for the function returns true.
        template <typename Predicate>
        [[nodiscard]] bool
        yield_while_timeout(Predicate&& predicate, std::chrono::duration<double> timeout,
            const char* thread_name = nullptr, bool allow_timed_suspension = true)
        {
            // Seconds represented using a double
            using duration_type = std::chrono::duration<double>;

            const bool use_timeout = timeout >= duration_type(0.0);
            auto yield_or_spin = allow_timed_suspension ?
                &pika::execution::this_thread::detail::yield_k :
                &pika::execution::this_thread::detail::spin_k;

            pika::chrono::detail::high_resolution_timer t;

            for (std::size_t k = 0;; ++k)
            {
                if (use_timeout && duration_type(t.elapsed()) > timeout) { return false; }

                if (!predicate()) { return true; }
                else { yield_or_spin(k, thread_name); }
            }
        }
    }    // namespace detail
}    // namespace pika::util
