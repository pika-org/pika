//  Copyright (c) 2024      ETH Zurich
//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2016      Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The implementation is based on the tree barrier from libcxx with the license below. Compared to the original:
// - Names have been de-uglified
// - Only the tree barrier has been kept (the original has an alternative non-tree implementation)
// - The heap allocation for the base implementation has been removed as it's not used here for ABI
//   stability
// - pika thread ids are used before std::thread ids are used for the tree index
// - Waiting is done with pika's yield_while, spinning until the expected result (yielding done
//   after some time)

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/concurrency/spinlock.hpp>
#include <pika/thread_support/assert_owns_lock.hpp>

#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <pika/config/warnings_prefix.hpp>

namespace pika {
    // The implementation is a classic tree barrier.

    // It looks different from literature pseudocode for two main reasons:
    //  1. Threads that call into std::barrier functions do not provide indices,
    //     so a numbering step is added before the actual barrier algorithm,
    //     appearing as an N+1 round to the N rounds of the tree barrier.
    //  2. A great deal of attention has been paid to avoid cache line thrashing
    //     by flattening the tree structure into cache-line sized arrays, that
    //     are indexed in an efficient way.
    namespace detail {
        struct empty_completion
        {
            void operator()() noexcept {}
        };

        using barrier_phase_t = std::uint8_t;

        class PIKA_EXPORT barrier_algorithm_base
        {
        public:
            // naturally-align the heap state
            struct alignas(64) state_t
            {
                struct
                {
                    std::atomic<detail::barrier_phase_t> phase{0};
                } tickets[64];
            };

            std::unique_ptr<state_t[]> state;

            barrier_algorithm_base(std::ptrdiff_t expected);
            bool arrive(std::ptrdiff_t expected, detail::barrier_phase_t old_phase);
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
    template <class Completion = detail::empty_completion>
    class barrier
    {
        std::ptrdiff_t expected;
        std::atomic<std::ptrdiff_t> expected_adjustment;
        std::decay_t<Completion> completion;
        std::atomic<detail::barrier_phase_t> phase;
        detail::barrier_algorithm_base base;

    public:
        using arrival_token = detail::barrier_phase_t;

        static constexpr std::ptrdiff_t max() noexcept
        {
            return std::numeric_limits<std::ptrdiff_t>::max();
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
        barrier(std::ptrdiff_t expected, Completion completion = Completion())
          : expected(expected)
          , expected_adjustment(0)
          , completion(std::move(completion))
          , phase(0)
          , base(this->expected)
        {
        }

        PIKA_NON_COPYABLE(barrier);

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
            auto const old_phase = phase.load(std::memory_order_relaxed);
            while (update != 0)
            {
                if (base.arrive(expected, old_phase))
                {
                    completion();
                    expected += expected_adjustment.load(std::memory_order_relaxed);
                    expected_adjustment.store(0, std::memory_order_relaxed);
                    phase.store(old_phase + 2, std::memory_order_release);
                }

                --update;
            }

            return old_phase;
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
            auto const poll = [&]() {
                // The original libcxx implementation uses the inverse condition here, since it
                // polls until the condition is true. Here we poll as long as the condition is true.
                return phase.load(std::memory_order_acquire) == old_phase;
            };

            bool const do_busy_wait = busy_wait_timeout > std::chrono::duration<double>(0.0);
            if (do_busy_wait &&
                pika::util::detail::yield_while_timeout(
                    poll, busy_wait_timeout, "barrier::wait", false))
            {
                return;
            }

            pika::util::yield_while(poll, "barrier::wait", true);
        }

        // Effects:        Equivalent to: wait(arrive()).
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
            expected_adjustment.fetch_sub(1, std::memory_order_relaxed);
            [[maybe_unused]] auto phase = arrive(1);
        }
    };
}    // namespace pika

#include <pika/config/warnings_suffix.hpp>
