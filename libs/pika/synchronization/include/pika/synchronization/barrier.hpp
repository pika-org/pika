//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  The algorithm was taken from http://locklessinc.com/articles/barriers/

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/concurrency/spinlock.hpp>
#include <pika/synchronization/detail/condition_variable.hpp>
#include <pika/thread_support/assert_owns_lock.hpp>

#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <pika/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace pika {

    namespace detail {

        struct empty_oncompletion
        {
            inline void operator()() noexcept {}
        };
    }    // namespace detail

    // A barrier is a thread coordination mechanism whose lifetime consists of
    // a sequence of barrier phases, where each phase allows at most an
    // expected number of threads to block until the expected number of threads
    // arrive at the barrier. [ Note: A barrier is useful for managing repeated
    // tasks that are handled by multiple threads. - end note ]

    // Each barrier phase consists of the following steps:
    //   - The expected count is decremented by each call to arrive or
    //     arrive_and_drop.
    //   - When the expected count reaches zero, the phase completion step is
    //     run. For the specialization with the default value of the
    //     CompletionFunction template parameter, the completion step is run
    //     as part of the call to arrive or arrive_and_drop that caused the
    //     expected count to reach zero. For other specializations, the
    //     completion step is run on one of the threads that arrived at the
    //     barrier during the phase.
    //   - When the completion step finishes, the expected count is reset to
    //     what was specified by the expected argument to the constructor,
    //     possibly adjusted by calls to arrive_and_drop, and the next phase
    //     starts.
    //
    // Each phase defines a phase synchronization point. Threads that arrive
    // at the barrier during the phase can block on the phase synchronization
    // point by calling wait, and will remain blocked until the phase
    // completion step is run.

    // The phase completion step that is executed at the end of each phase has
    // the following effects:
    //   - Invokes the completion function, equivalent to completion().
    //   - Unblocks all threads that are blocked on the phase synchronization
    //     point.
    // The end of the completion step strongly happens before the returns from
    // all calls that were unblocked by the completion step. For
    // specializations that do not have the default value of the
    // CompletionFunction template parameter, the behavior is undefined if any
    // of the barrier object's member functions other than wait are called
    // while the completion step is in progress.
    //
    // Concurrent invocations of the member functions of barrier, other than
    // its destructor, do not introduce data races. The member functions
    // arrive and arrive_and_drop execute atomically.
    //
    // CompletionFunction shall meet the Cpp17MoveConstructible (Table 28) and
    // Cpp17Destructible (Table 32) requirements.
    // std::is_nothrow_invocable_v<CompletionFunction&> shall be true.
    //
    // The default value of the CompletionFunction template parameter is an
    // unspecified type, such that, in addition to satisfying the requirements
    // of CompletionFunction, it meets the Cpp17DefaultConstructible
    // requirements (Table 27) and completion() has no effects.
    //
    // barrier::arrival_token is an unspecified type, such that it meets the
    // Cpp17MoveConstructible (Table 28), Cpp17MoveAssignable (Table 30), and
    // Cpp17Destructible (Table 32) requirements.

    template <typename OnCompletion = detail::empty_oncompletion>
    class barrier
    {
    public:
        PIKA_NON_COPYABLE(barrier);

    private:
        using mutex_type = pika::concurrency::detail::spinlock;

    public:
        using arrival_token = bool;

        // Returns:        The maximum expected count that the implementation
        //                 supports.
        static constexpr std::ptrdiff_t(max)() noexcept
        {
            return (std::numeric_limits<std::ptrdiff_t>::max)();
        }

        // Preconditions:  expected >= 0 is true and expected <= max() is true.
        // Effects:        Sets both the initial expected count for each
        //                 barrier phase and the current expected count for the
        //                 first phase to expected. Initializes completion with
        //                 PIKA_MOVE(f). Starts the first phase. [Note: If
        //                 expected is 0 this object can only be destroyed.-
        //                 end note]
        // Throws:         Any exception thrown by CompletionFunction's move
        //                 constructor.
        barrier(std::ptrdiff_t expected, OnCompletion completion = OnCompletion())
          : expected_(expected)
          , arrived_(expected)
          , completion_(PIKA_MOVE(completion))
          , phase_(false)
        {
            PIKA_ASSERT(expected >= 0 && expected <= (max) ());
        }

    private:
        [[nodiscard]] arrival_token arrive_locked(
            std::unique_lock<mutex_type>& l, std::ptrdiff_t update = 1)
        {
            PIKA_ASSERT_OWNS_LOCK(l);
            PIKA_ASSERT(arrived_ >= update);

            bool const old_phase = phase_;
            std::ptrdiff_t const result = (arrived_ -= update);
            std::ptrdiff_t const new_expected = expected_;
            if (result == 0)
            {
                completion_();
                arrived_ = new_expected;
                phase_ = !old_phase;
                cond_.notify_all(PIKA_MOVE(l));
            }
            return old_phase;
        }

    public:
        // Preconditions:  update > 0 is true, and update is less than or equal
        //                 to the expected count for the current barrier phase.
        // Effects:        Constructs an object of type arrival_token that is
        //                 associated with the phase synchronization point for
        //                 the current phase. Then, decrements the expected
        //                 count by update.
        // Synchronization: The call to arrive strongly happens before the
        //                 start of the phase completion step for the current
        //                 phase.
        // Returns:        The constructed arrival_token object.
        // Throws:         system_error when an exception is required
        //                 ([thread.req.exception]).
        // Error conditions: Any of the error conditions allowed for mutex
        //                 types([thread.mutex.requirements.mutex]).
        // [Note: This call can cause the completion step for the current phase
        //        to start.- end note]
        [[nodiscard]] arrival_token arrive(std::ptrdiff_t update = 1)
        {
            std::unique_lock<mutex_type> l(mtx_);
            return arrive_locked(l, update);
        }

        // Preconditions:  arrival is associated with the phase synchronization
        //                 point for the current phase or the immediately
        //                 preceding phase of the same barrier object.
        // Effects:        Blocks at the synchronization point associated with
        //                 PIKA_MOVE(arrival) until the phase completion step
        //                 of the synchronization point's phase is run. [ Note:
        //                 If arrival is associated with the synchronization
        //                 point for a previous phase, the call returns
        //                 immediately. - end note ]
        // Throws:         system_error when an exception is required
        //                 ([thread.req.exception]).
        // Error conditions: Any of the error conditions allowed for mutex
        //                 types ([thread.mutex.requirements.mutex]).
        void wait(arrival_token&& old_phase,
            std::chrono::duration<double> busy_wait_timeout = std::chrono::duration<double>(
                0.0)) const
        {
            bool const do_busy_wait = busy_wait_timeout > std::chrono::duration<double>(0.0);
            if (do_busy_wait &&
                pika::util::detail::yield_while_timeout(
                    [&]() {
                        std::unique_lock<mutex_type> l(mtx_);
                        return phase_ == old_phase;
                    },
                    busy_wait_timeout, "barrier::wait", false))
            {
                return;
            }

            std::unique_lock<mutex_type> l(mtx_);
            if (phase_ == old_phase)
            {
                cond_.wait(l, "barrier::wait");
            }
            PIKA_ASSERT(phase_ != old_phase);
        }

        /// Effects:        Equivalent to: wait(arrive()).
        void arrive_and_wait(
            std::chrono::duration<double> busy_wait_timeout = std::chrono::duration<double>(0.0))
        {
            wait(arrive(), busy_wait_timeout);
        }

        // Preconditions:  The expected count for the current barrier phase is
        //                 greater than zero.
        // Effects:        Decrements the initial expected count for all
        //                 subsequent phases by one. Then decrements the
        //                 expected count for the current phase by one.
        // Synchronization: The call to arrive_and_drop strongly happens before
        //                 the start of the phase completion step for the
        //                 current phase.
        // Throws:         system_error when an exception is required
        //                 ([thread.req.exception]).
        // Error conditions: Any of the error conditions allowed for mutex
        //                 types ([thread.mutex.requirements.mutex]).
        // [Note: This call can cause the completion step for the current
        //        phase to start.- end note]
        void arrive_and_drop()
        {
            std::unique_lock<mutex_type> l(mtx_);
            PIKA_ASSERT(expected_ > 0);
            --expected_;
            PIKA_UNUSED(arrive_locked(l, 1));
        }

    private:
        mutable mutex_type mtx_;
        mutable pika::detail::condition_variable cond_;

        std::ptrdiff_t expected_;
        std::ptrdiff_t arrived_;
        OnCompletion completion_;
        bool phase_;
    };

}    // namespace pika

#include <pika/config/warnings_suffix.hpp>
