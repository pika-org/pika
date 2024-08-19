//  Copyright (c) 2024      ETH Zurich
//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2016      Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The implementation is based on the tree barrier from libc++ with the license below. See header
// file for differences to the original.

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

PIKA_GLOBAL_MODULE_FRAGMENT

#if !defined(PIKA_HAVE_MODULE)
#include <pika/synchronization/barrier.hpp>
#include <pika/threading_base/thread_data.hpp>
#endif

#include <atomic>
#include <cstddef>
#include <memory>
#include <thread>

#if defined(PIKA_HAVE_MODULE)
module pika.synchronization;
#endif

namespace pika::detail {
    barrier_algorithm_base::barrier_algorithm_base(std::ptrdiff_t expected)
    {
        std::size_t const count = (expected + 1) >> 1;
        state = std::unique_ptr<state_t[]>(new state_t[count]);
    }

    bool barrier_algorithm_base::arrive(std::ptrdiff_t expected, detail::barrier_phase_t old_phase)
    {
        detail::barrier_phase_t const half_step = old_phase + 1, full_step = old_phase + 2;
        std::size_t current_expected = expected;
        auto pika_thread_id = pika::threads::detail::get_self_id();

        // The original libc++ implementation uses only the id of the current std::thread as the
        // input for the hash. This implementation prefers to use the pika thread id if available,
        // and otherwise uses the std::thread id.
        std::size_t current = pika_thread_id == pika::threads::detail::invalid_thread_id ?
            std::hash<pika::threads::detail::thread_id_type>()(
                pika::threads::detail::get_self_id()) :
            std::hash<std::thread::id>()(std::this_thread::get_id()) % ((expected + 1) >> 1);
        for (int round = 0;; ++round)
        {
            if (current_expected <= 1) { return true; }

            std::size_t const end_node = ((current_expected + 1) >> 1), last_node = end_node - 1;

            while (true)
            {
                if (current == end_node) current = 0;
                detail::barrier_phase_t expect = old_phase;
                if (current == last_node && (current_expected & 1))
                {
                    if (state[current].tickets[round].phase.compare_exchange_strong(
                            expect, full_step, std::memory_order_acq_rel))
                        break;    // I'm 1 in 1, go to next round
                }
                else if (state[current].tickets[round].phase.compare_exchange_strong(
                             expect, half_step, std::memory_order_acq_rel))
                {
                    return false;    // I'm 1 in 2, done with arrival
                }
                else if (expect == half_step)
                {
                    if (state[current].tickets[round].phase.compare_exchange_strong(
                            expect, full_step, std::memory_order_acq_rel))
                        break;    // I'm 2 in 2, go to next round
                }

                ++current;
            }

            current_expected = last_node + 1;
            current >>= 1;
        }
    }
}    // namespace pika::detail
