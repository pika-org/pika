//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/init.hpp>

#include <fmt/printf.h>
#include <fmt/ranges.h>

#include <cstddef>
#include <random>
#include <utility>
#include <vector>

std::size_t get_n() { return 13; }
std::size_t calculate(std::size_t i) { return (std::rand() % 4) * i * i; }

int main(int argc, char* argv[])
{
    namespace ex = pika::execution::experimental;
    namespace tt = pika::this_thread::experimental;

    pika::start(argc, argv);
    ex::thread_pool_scheduler sched{};

    // when_all_vector is like when_all, but for a dynamic number of senders
    // through a vector of senders
    auto const n = get_n();
    std::vector<ex::unique_any_sender<std::size_t>> snds;
    snds.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        snds.push_back(ex::just(i) | ex::continues_on(sched) | ex::then(calculate));
    }
    auto snds_print =
        ex::when_all_vector(std::move(snds)) | ex::then([](std::vector<std::size_t> results) {
            fmt::print("Results are: {}\n", fmt::join(results, ", "));
        });
    tt::sync_wait(std::move(snds_print));

    // when_all_vector will send no value on completion if the input vector
    // contains senders sending no value
    std::vector<ex::unique_any_sender<>> snds_nothing;
    snds_nothing.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        snds_nothing.push_back(ex::just(i) | ex::continues_on(sched) |
            ex::then([](auto i) { fmt::print("{}: {}\n", i, calculate(i)); }));
    }
    auto snds_nothing_done = ex::when_all_vector(std::move(snds_nothing)) |
        ex::then([]() { fmt::print("Done printing all results\n"); });
    tt::sync_wait(std::move(snds_nothing_done));

    pika::finalize();
    pika::stop();

    return 0;
}
