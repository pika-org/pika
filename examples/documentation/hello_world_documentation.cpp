//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/thread.hpp>

#include <fmt/printf.h>

#include <utility>

int main(int argc, char* argv[])
{
    // Most functionality is found in the pika::execution namespace. If pika is built with stdexec,
    // std::execution will also be found in this namespace.
    namespace ex = pika::execution::experimental;
    // Some additional utilities are in pika::this_thread.
    namespace tt = pika::this_thread::experimental;

    // Start the pika runtime.
    pika::start(argc, argv);

    // Create a std::execution scheduler that runs work on the default pika thread pool.
    ex::thread_pool_scheduler sched{};

    // We can schedule work using sched.
    auto snd1 = ex::just(42) | ex::continues_on(sched) | ex::then([](int x) {
        fmt::print("Hello from a pika user-level thread (with id {})!\nx = {}\n",
            pika::this_thread::get_id(), x);
    });

    // The work is started once we call sync_wait.
    tt::sync_wait(std::move(snd1));

    // We can build arbitrary graphs of work using the split and when_all adaptors.
    auto snd2 = ex::just(3.14) | ex::split();
    auto snd3 = ex::continues_on(snd2, sched) |
        ex::then([](double pi) { fmt::print("Is this pi: {}?\n", pi); });
    auto snd4 = ex::when_all(std::move(snd2), ex::just(500.3)) | ex::continues_on(sched) |
        ex::then([](double pi, double r) { return pi * r * r; });
    auto result = tt::sync_wait(ex::when_all(std::move(snd3), std::move(snd4)));
    fmt::print("The result is {}\n", result);

    // Tell the runtime that when there are no more tasks in the queues it is ok to stop.
    pika::finalize();

    // Wait for all work to finish and stop the runtime.
    pika::stop();

    return 0;
}
