//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/init.hpp>

#include <fmt/printf.h>

#include <chrono>
#include <thread>
#include <tuple>
#include <utility>

int main(int argc, char* argv[])
{
    namespace ex = pika::execution::experimental;
    namespace tt = pika::this_thread::experimental;

    pika::start(argc, argv);
    ex::thread_pool_scheduler sched{};

    // split_tuple can be used to process the result and its square through
    // senders, without having to pass both around together
    auto [snd, snd_squared] = ex::schedule(sched) | ex::then([]() { return 42; }) |
        ex::then([](int x) { return std::tuple(x, x * x); }) | ex::split_tuple();

    // snd and snd_squared will be ready at the same time, but can be used independently
    auto snd_print = std::move(snd) | ex::continues_on(sched) |
        ex::then([](int x) { fmt::print("x is {}\n", x); });
    auto snd_process =
        std::move(snd_squared) | ex::continues_on(sched) | ex::then([](int x_squared) {
            fmt::print("Performing expensive operations on x * x\n");
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            return x_squared / 2;
        });

    auto x_squared_processed =
        tt::sync_wait(ex::when_all(std::move(snd_print), std::move(snd_process)));
    fmt::print("The final result is {}\n", x_squared_processed);

    pika::finalize();
    pika::stop();

    return 0;
}
