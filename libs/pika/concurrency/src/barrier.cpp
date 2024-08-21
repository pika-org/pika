//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

PIKA_GLOBAL_MODULE_FRAGMENT

#include <pika/config.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/concurrency/barrier.hpp>

#include <cstddef>
#include <mutex>
#endif

#if defined(PIKA_HAVE_MODULE)
module pika.concurrency;
#endif

namespace pika::concurrency::detail {
    barrier::barrier(std::size_t number_of_threads)
      : number_of_threads_(number_of_threads)
      , total_(barrier_flag)
      , mtx_()
      , cond_()
    {
    }

    barrier::~barrier()
    {
        std::unique_lock<mutex_type> l(mtx_);

        // Wait until everyone exits the barrier
        cond_.wait(l, [&] { return total_ <= barrier_flag; });
    }

    void barrier::wait()
    {
        std::unique_lock<mutex_type> l(mtx_);

        // Wait until everyone exits the barrier
        cond_.wait(l, [&] { return total_ <= barrier_flag; });

        // Are we the first to enter?
        if (total_ == barrier_flag) total_ = 0;

        ++total_;

        if (total_ == number_of_threads_)
        {
            total_ += barrier_flag - 1;
            cond_.notify_all();
        }
        else
        {
            // Wait until enough threads enter the barrier
            cond_.wait(l, [&] { return total_ >= barrier_flag; });

            --total_;

            // get entering threads to wake up
            if (total_ == barrier_flag) { cond_.notify_all(); }
        }
    }
}    // namespace pika::concurrency::detail
