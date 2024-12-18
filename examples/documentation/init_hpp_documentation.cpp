//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/init.hpp>

#include <fmt/printf.h>

#include <utility>

int main(int argc, char* argv[])
{
    namespace ex = pika::execution::experimental;
    namespace tt = pika::this_thread::experimental;

    pika::start(argc, argv);

    // The pika runtime is now active and we can schedule work on the default
    // thread pool
    auto s = ex::schedule(ex::thread_pool_scheduler{}) |
        ex::then([]() { fmt::print("Hello from the pika runtime\n"); });
    tt::sync_wait(std::move(s));

    pika::finalize();
    pika::stop();

    return 0;
}
