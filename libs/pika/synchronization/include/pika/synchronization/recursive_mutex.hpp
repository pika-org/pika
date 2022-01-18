//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Part of this code has been adopted from code published under the BSL by:
//
//  (C) Copyright 2006-7 Anthony Williams
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/execution_base/agent_ref.hpp>
#include <pika/execution_base/this_thread.hpp>
#include <pika/synchronization/spinlock.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace lcos { namespace local {
    namespace detail {
        /// An exclusive-ownership recursive mutex which implements Boost.Thread's
        /// TimedLockable concept.
        template <typename Mutex = local::spinlock>
        struct recursive_mutex_impl
        {
        public:
            PIKA_NON_COPYABLE(recursive_mutex_impl);

        private:
            std::atomic<std::uint64_t> recursion_count;
            std::atomic<pika::execution_base::agent_ref> locking_context;
            Mutex mtx;

        public:
            recursive_mutex_impl(
                char const* const desc = "recursive_mutex_impl")
              : recursion_count(0)
              , mtx(desc)
            {
            }

            /// Attempts to acquire ownership of the \a recursive_mutex.
            /// Never blocks.
            ///
            /// \returns \a true if ownership was acquired; otherwise, \a false.
            ///
            /// \throws Never throws.
            bool try_lock()
            {
                auto ctx = pika::execution_base::this_thread::agent();
                PIKA_ASSERT(ctx);

                return try_recursive_lock(ctx) || try_basic_lock(ctx);
            }

            /// Acquires ownership of the \a recursive_mutex. Suspends the
            /// current pika-thread if ownership cannot be obtained immediately.
            ///
            /// \throws Throws \a pika#bad_parameter if an error occurs while
            ///         suspending. Throws \a pika#yield_aborted if the mutex is
            ///         destroyed while suspended. Throws \a pika#null_thread_id if
            ///         called outside of a pika-thread.
            void lock()
            {
                auto ctx = pika::execution_base::this_thread::agent();
                PIKA_ASSERT(ctx);

                if (!try_recursive_lock(ctx))
                {
                    mtx.lock();
                    locking_context.exchange(ctx);
                    util::ignore_lock(&mtx);
                    util::register_lock(this);
                    recursion_count.store(1);
                }
            }

            /// Release ownership of the \a recursive_mutex.
            ///
            /// \throws Throws \a pika#bad_parameter if an error occurs while
            ///         releasing the mutex. Throws \a pika#null_thread_id if called
            ///         outside of a pika-thread.
            void unlock()
            {
                if (0 == --recursion_count)
                {
                    locking_context.exchange(pika::execution_base::agent_ref());
                    util::unregister_lock(this);
                    util::reset_ignored(&mtx);
                    mtx.unlock();
                }
            }

        private:
            bool try_recursive_lock(
                pika::execution_base::agent_ref current_context)
            {
                if (locking_context.load(std::memory_order_acquire) ==
                    current_context)
                {
                    if (++recursion_count == 1)
                        util::register_lock(this);
                    return true;
                }
                return false;
            }

            bool try_basic_lock(pika::execution_base::agent_ref current_context)
            {
                if (mtx.try_lock())
                {
                    locking_context.exchange(current_context);
                    util::ignore_lock(&mtx);
                    util::register_lock(this);
                    recursion_count.store(1);
                    return true;
                }
                return false;
            }
        };
    }    // namespace detail

    using recursive_mutex = detail::recursive_mutex_impl<>;
}}}    // namespace pika::lcos::local
