//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>

#include <chrono>

// This test ensures that thread creation uses the correct stack sizes. We
// slightly change all the stack sizes in the configuration to catch problems
// with the used stack sizes not matching the configured sizes.

int pika_main()
{
    pika::this_thread::yield();

    pika::thread t([]() {});
    t.join();

    return pika::local::finalize();
}

int main(int argc, char** argv)
{
    pika::local::init_params p;
    p.cfg = {"pika.stacks.small_size=" +
            std::to_string(PIKA_SMALL_STACK_SIZE + 0x1000),
        "pika.stacks.medium_size=" +
            std::to_string(PIKA_MEDIUM_STACK_SIZE + 0x1000),
        "pika.stacks.large_size=" +
            std::to_string(PIKA_LARGE_STACK_SIZE + 0x1000),
        "pika.stacks.huge_size=" + std::to_string(PIKA_HUGE_STACK_SIZE + 0x1000)};

    return pika::local::init(pika_main, argc, argv, p);
}
