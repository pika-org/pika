//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/init.hpp>

#include <fmt/printf.h>

int main(int argc, char* argv[])
{
    pika::start(argc, argv);

    // The pika runtime is now active and we can schedule work on the default thread pool
    pika::this_thread::experimental::sync_wait(
        pika::execution::experimental::schedule(
            pika::execution::experimental::thread_pool_scheduler{}) |
        pika::execution::experimental::then([]() { fmt::print("Hello from the pika runtime\n"); }));

    pika::finalize();
    pika::stop();

    return 0;
}
