//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/concurrency/spinlock.hpp>
#include <pika/synchronization/detail/counting_semaphore.hpp>
#include <pika/timing/steady_clock.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>

#if defined(PIKA_MSVC_WARNING_PRAGMA)
# pragma warning(push)
# pragma warning(disable : 4251)
#endif

////////////////////////////////////////////////////////////////////////////////
namespace pika {
    // A semaphore is a protected variable (an entity storing a value) or
    // abstract data type (an entity grouping several variables that may or
    // may not be numerical) which constitutes the classic method for
    // restricting access to shared resources, such as shared memory, in a
    // multiprogramming environment. Semaphores exist in many variants, though
    // usually the term refers to a counting semaphore, since a binary
    // semaphore is better known as a mutex. A counting semaphore is a counter
    // for a set of available resources, rather than a locked/unlocked flag of
    // a single resource. It was invented by Edsger Dijkstra. Semaphores are
    // the classic solution to preventing race conditions in the dining
    // philosophers problem, although they do not prevent resource deadlocks.
    //
    // Counting semaphores can be used for synchronizing multiple threads as
    // well: one thread waiting for several other threads to touch (signal)
    // the semaphore, or several threads waiting for one other thread to touch
    // this semaphore.
    template <std::ptrdiff_t LeastMaxValue = PTRDIFF_MAX,
        typename Mutex = pika::concurrency::detail::spinlock>
    class counting_semaphore
    {
    public:
        PIKA_NON_COPYABLE(counting_semaphore);

    protected:
        using mutex_type = Mutex;

    public:
        // Returns The maximum value of counter. This value is greater than or
        // equal to LeastMaxValue.
        static constexpr std::ptrdiff_t(max)() noexcept
        {
            return LeastMaxValue;
        }

        // \brief Construct a new counting semaphore
        //
        // \param value    [in] The initial value of the internal semaphore
        //                 lock count. Normally this value should be zero
        //                 (which is the default), values greater than zero
        //                 are equivalent to the same number of signals pre-
        //                 set, and negative values are equivalent to the
        //                 same number of waits pre-set.
        //
        //  Preconditions   value >= 0 is true, and value <= max() is true.
        //  Effects         Initializes counter with desired.
        //  Throws          Nothing.
        explicit counting_semaphore(std::ptrdiff_t value)
          : sem_(value)
        {
        }

        ~counting_semaphore() = default;

        // Preconditions:   update >= 0 is true, and update <= max() - counter
        //                  is true.
        // Effects:         Atomically execute counter += update. Then, unblocks
        //                  any threads that are waiting for counter to be
        //                  greater than zero.
        // Synchronization: Strongly happens before invocations of try_acquire
        //                  that observe the result of the effects.
        // Throws:          system_error when an exception is required
        //                  ([thread.req.exception]).
        // Error conditions: Any of the error conditions allowed for mutex
        //                  types ([thread.mutex.requirements.mutex]).
        void release(std::ptrdiff_t update = 1)
        {
            std::unique_lock<mutex_type> l(mtx_);
            sem_.signal(PIKA_MOVE(l), update);
        }

        // Effects:         Attempts to atomically decrement counter if it is
        //                  positive, without blocking. If counter is not
        //                  decremented, there is no effect and try_acquire
        //                  immediately returns. An implementation may fail to
        //                  decrement counter even if it is positive. [ Note:
        //                  This spurious failure is normally uncommon, but
        //                  allows interesting implementations based on a simple
        //                  compare and exchange ([atomics]). - end note ]
        //                  An implementation should ensure that try_acquire
        //                  does not consistently return false in the absence
        //                  of contending semaphore operations.
        // Returns:         true if counter was decremented, otherwise false.
        bool try_acquire() noexcept
        {
            std::unique_lock<mutex_type> l(mtx_);
            return sem_.try_acquire(l);
        }

        // Effects:         Repeatedly performs the following steps, in order:
        //                    - Evaluates try_acquire. If the result is true,
        //                      returns.
        //                    - Blocks on *this until counter is greater than
        //                      zero.
        // Throws:          system_error when an exception is required
        //                  ([thread.req.exception]).
        // Error conditions: Any of the error conditions allowed for mutex
        //                  types ([thread.mutex.requirements.mutex]).
        void acquire()
        {
            std::unique_lock<mutex_type> l(mtx_);
            sem_.wait(l, 1);
        }

        // Effects:         Repeatedly performs the following steps, in order:
        //                    - Evaluates try_acquire(). If the result is true,
        //                      returns true.
        //                    - Blocks on *this until counter is greater than
        //                      zero or until the timeout expires. If it is
        //                      unblocked by the timeout expiring, returns false.
        //                  The timeout expires ([thread.req.timing]) when the
        //                  current time is after abs_time (for
        //                  try_acquire_until) or when at least rel_time has
        //                  passed from the start of the function (for
        //                  try_acquire_for).
        // Throws:          Timeout - related exceptions ([thread.req.timing]),
        //                  or system_error when a non - timeout - related
        //                  exception is required ([thread.req.exception]).
        // Error conditions: Any of the error conditions allowed for mutex types
        //                  ([thread.mutex.requirements.mutex]).
        bool try_acquire_until(pika::chrono::steady_time_point const& abs_time)
        {
            std::unique_lock<mutex_type> l(mtx_);
            return sem_.wait_until(l, abs_time, 1);
        }

        bool try_acquire_for(pika::chrono::steady_duration const& rel_time)
        {
            return try_acquire_until(rel_time.from_now());
        }

    protected:
        mutable mutex_type mtx_;
        detail::counting_semaphore sem_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex = pika::concurrency::detail::spinlock>
    class binary_semaphore : public counting_semaphore<1, Mutex>
    {
    public:
        PIKA_NON_COPYABLE(binary_semaphore);

    public:
        binary_semaphore(std::ptrdiff_t value = 1)
          : counting_semaphore<1, Mutex>(value)
        {
        }

        ~binary_semaphore() = default;
    };

}    // namespace pika

#if defined(PIKA_MSVC_WARNING_PRAGMA)
# pragma warning(pop)
#endif
