//  Copyright (c) 2016-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/datastructures/optional.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/functional/detail/invoke.hpp>
#include <pika/functional/invoke_result.hpp>
#include <pika/synchronization/spinlock.hpp>
#include <pika/threading_base/thread_helpers.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>

namespace pika { namespace threads {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        // This is the overload for running functions which return a value.
        template <typename F, typename... Ts>
        typename util::invoke_result<F, Ts...>::type run_as_pika_thread(
            std::false_type, F const& f, Ts&&... ts)
        {
            // NOTE: The condition variable needs be able to live past the scope
            // of this function. The mutex and boolean are guaranteed to live
            // long enough because of the lock.
            pika::lcos::local::spinlock mtx;
            auto cond = std::make_shared<std::condition_variable_any>();
            bool stopping = false;

            typedef typename util::invoke_result<F, Ts...>::type result_type;

            // Using the optional for storing the returned result value
            // allows to support non-default-constructible and move-only
            // types.
            pika::util::optional<result_type> result;
            std::exception_ptr exception;

            // Create the pika thread
            pika::threads::thread_init_data data(
                pika::threads::make_thread_function_nullary([&, cond]() {
                    try
                    {
                        // Execute the given function, forward all parameters,
                        // store result.
                        result.emplace(PIKA_INVOKE(f, PIKA_FORWARD(Ts, ts)...));
                    }
                    catch (...)
                    {
                        // make sure exceptions do not escape the pika thread
                        // scheduler
                        exception = std::current_exception();
                    }

                    // Now signal to the waiting thread that we're done.
                    {
                        std::lock_guard<pika::lcos::local::spinlock> lk(mtx);
                        stopping = true;
                    }
                    cond->notify_all();
                }),
                "run_as_pika_thread (non-void)");
            pika::threads::register_work(data);

            // wait for the pika thread to exit
            std::unique_lock<pika::lcos::local::spinlock> lk(mtx);
            cond->wait(lk, [&]() -> bool { return stopping; });

            // rethrow exceptions
            if (exception)
                std::rethrow_exception(exception);

            return PIKA_MOVE(*result);
        }

        // This is the overload for running functions which return void.
        template <typename F, typename... Ts>
        void run_as_pika_thread(std::true_type, F const& f, Ts&&... ts)
        {
            // NOTE: The condition variable needs be able to live past the scope
            // of this function. The mutex and boolean are guaranteed to live
            // long enough because of the lock.
            pika::lcos::local::spinlock mtx;
            auto cond = std::make_shared<std::condition_variable_any>();
            bool stopping = false;

            std::exception_ptr exception;

            // Create an pika thread
            pika::threads::thread_init_data data(
                pika::threads::make_thread_function_nullary([&, cond]() {
                    try
                    {
                        // Execute the given function, forward all parameters.
                        PIKA_INVOKE(f, PIKA_FORWARD(Ts, ts)...);
                    }
                    catch (...)
                    {
                        // make sure exceptions do not escape the pika thread
                        // scheduler
                        exception = std::current_exception();
                    }

                    // Now signal to the waiting thread that we're done.
                    {
                        std::lock_guard<pika::lcos::local::spinlock> lk(mtx);
                        stopping = true;
                    }
                    cond->notify_all();
                }),
                "run_as_pika_thread (void)");
            pika::threads::register_work(data);

            // wait for the pika thread to exit
            std::unique_lock<pika::lcos::local::spinlock> lk(mtx);
            cond->wait(lk, [&]() -> bool { return stopping; });

            // rethrow exceptions
            if (exception)
                std::rethrow_exception(exception);
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    typename util::invoke_result<F, Ts...>::type run_as_pika_thread(
        F const& f, Ts&&... vs)
    {
        // This shouldn't be used on a pika-thread
        PIKA_ASSERT(pika::threads::get_self_ptr() == nullptr);

        typedef typename std::is_void<
            typename util::invoke_result<F, Ts...>::type>::type result_is_void;

        return detail::run_as_pika_thread(
            result_is_void(), f, PIKA_FORWARD(Ts, vs)...);
    }
}}    // namespace pika::threads
