//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/condition_variable.hpp>
#include <pika/local/exception.hpp>
#include <pika/local/functional.hpp>
#include <pika/local/init.hpp>
#include <pika/local/mutex.hpp>
#include <pika/modules/testing.hpp>
#include <pika/runtime_local/run_as_pika_thread.hpp>

#include <functional>
#include <mutex>
#include <thread>

std::mutex startup_mtx;
std::condition_variable startup_cond;

bool running = false;
bool stop_running = false;

int start_func(pika::lcos::local::spinlock& mtx,
    pika::lcos::local::condition_variable_any& cond)
{
    // Signal to constructor that thread has started running.
    {
        std::lock_guard<std::mutex> lk(startup_mtx);
        running = true;
    }

    {
        std::unique_lock<pika::lcos::local::spinlock> lk(mtx);
        startup_cond.notify_one();
        while (!stop_running)
            cond.wait(lk);
    }

    return pika::local::finalize();
}

void pika_thread_func()
{
    PIKA_THROW_EXCEPTION(pika::invalid_status, "pika_thread_func", "test");
}

int main(int argc, char** argv)
{
    pika::lcos::local::spinlock mtx;
    pika::lcos::local::condition_variable_any cond;

    pika::util::function_nonser<int(int, char**)> start_function =
        pika::util::bind(&start_func, std::ref(mtx), std::ref(cond));

    pika::local::start(start_function, argc, argv);

    // wait for the main pika thread to run
    {
        std::unique_lock<std::mutex> lk(startup_mtx);
        while (!running)
            startup_cond.wait(lk);
    }

    bool exception_caught = false;
    try
    {
        pika::threads::run_as_pika_thread(&pika_thread_func);
        PIKA_TEST(false);    // this should not be executed
    }
    catch (...)
    {
        exception_caught = true;
    }
    PIKA_TEST(exception_caught);

    {
        std::lock_guard<pika::lcos::local::spinlock> lk(mtx);
        stop_running = true;
    }

    cond.notify_one();

    return pika::local::stop();
}
