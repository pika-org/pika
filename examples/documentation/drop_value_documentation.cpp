//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/init.hpp>

#include <fmt/printf.h>

#include <tuple>
#include <utility>

struct custom_type
{
};

int main(int argc, char* argv[])
{
    namespace ex = pika::execution::experimental;
    namespace tt = pika::this_thread::experimental;

    pika::start(argc, argv);
    ex::thread_pool_scheduler sched{};

    auto s = ex::just(42, custom_type{}, std::tuple("hello")) | ex::drop_value() |
        // No matter what is sent to drop_value, it won't be sent from drop_value
        ex::then([] { fmt::print("I got nothing...\n"); });
    tt::sync_wait(std::move(s));

    pika::finalize();
    pika::stop();

    return 0;
}
