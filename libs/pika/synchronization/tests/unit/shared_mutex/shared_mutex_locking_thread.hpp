//  (C) Copyright 2008 Anthony Williams
//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/synchronization/condition_variable.hpp>
#include <pika/synchronization/mutex.hpp>
#include <pika/synchronization/shared_mutex.hpp>

#include <mutex>
#include <shared_mutex>

namespace test {
    template <typename Lock>
    class locking_thread
    {
    private:
        pika::lcos::local::shared_mutex& rw_mutex;
        unsigned& unblocked_count;
        pika::lcos::local::condition_variable& unblocked_condition;
        unsigned& simultaneous_running_count;
        unsigned& max_simultaneous_running;
        pika::lcos::local::mutex& unblocked_count_mutex;
        pika::lcos::local::mutex& finish_mutex;

    public:
        locking_thread(pika::lcos::local::shared_mutex& rw_mutex_,
            unsigned& unblocked_count_,
            pika::lcos::local::mutex& unblocked_count_mutex_,
            pika::lcos::local::condition_variable& unblocked_condition_,
            pika::lcos::local::mutex& finish_mutex_,
            unsigned& simultaneous_running_count_,
            unsigned& max_simultaneous_running_)
          : rw_mutex(rw_mutex_)
          , unblocked_count(unblocked_count_)
          , unblocked_condition(unblocked_condition_)
          , simultaneous_running_count(simultaneous_running_count_)
          , max_simultaneous_running(max_simultaneous_running_)
          , unblocked_count_mutex(unblocked_count_mutex_)
          , finish_mutex(finish_mutex_)
        {
        }

        void operator()()
        {
            // acquire lock
            Lock lock(rw_mutex);

            // increment count to show we're unblocked
            {
                std::unique_lock<pika::lcos::local::mutex> ublock(
                    unblocked_count_mutex);

                ++unblocked_count;
                unblocked_condition.notify_one();
                ++simultaneous_running_count;
                if (simultaneous_running_count > max_simultaneous_running)
                {
                    max_simultaneous_running = simultaneous_running_count;
                }
            }

            // wait to finish
            std::unique_lock<pika::lcos::local::mutex> finish_lock(finish_mutex);
            {
                std::unique_lock<pika::lcos::local::mutex> ublock(
                    unblocked_count_mutex);

                --simultaneous_running_count;
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    class simple_writing_thread
    {
    private:
        pika::lcos::local::shared_mutex& rwm;
        pika::lcos::local::mutex& finish_mutex;
        pika::lcos::local::mutex& unblocked_mutex;
        unsigned& unblocked_count;

    public:
        simple_writing_thread(pika::lcos::local::shared_mutex& rwm_,
            pika::lcos::local::mutex& finish_mutex_,
            pika::lcos::local::mutex& unblocked_mutex_,
            unsigned& unblocked_count_)
          : rwm(rwm_)
          , finish_mutex(finish_mutex_)
          , unblocked_mutex(unblocked_mutex_)
          , unblocked_count(unblocked_count_)
        {
        }

        void operator()()
        {
            std::unique_lock<pika::lcos::local::shared_mutex> lk(rwm);
            {
                std::unique_lock<pika::lcos::local::mutex> ulk(unblocked_mutex);
                ++unblocked_count;
            }
            std::unique_lock<pika::lcos::local::mutex> flk(finish_mutex);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    class simple_reading_thread
    {
    private:
        pika::lcos::local::shared_mutex& rwm;
        pika::lcos::local::mutex& finish_mutex;
        pika::lcos::local::mutex& unblocked_mutex;
        unsigned& unblocked_count;

    public:
        simple_reading_thread(pika::lcos::local::shared_mutex& rwm_,
            pika::lcos::local::mutex& finish_mutex_,
            pika::lcos::local::mutex& unblocked_mutex_,
            unsigned& unblocked_count_)
          : rwm(rwm_)
          , finish_mutex(finish_mutex_)
          , unblocked_mutex(unblocked_mutex_)
          , unblocked_count(unblocked_count_)
        {
        }

        void operator()()
        {
            std::shared_lock<pika::lcos::local::shared_mutex> lk(rwm);
            {
                std::unique_lock<pika::lcos::local::mutex> ulk(unblocked_mutex);
                ++unblocked_count;
            }
            std::unique_lock<pika::lcos::local::mutex> flk(finish_mutex);
        }
    };
}    // namespace test
