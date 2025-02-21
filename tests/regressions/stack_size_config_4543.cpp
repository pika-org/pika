//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/init.hpp>
#include <pika/testing.hpp>
#include <pika/thread.hpp>

#include <chrono>
#include <cstdlib>

// This test ensures that thread creation uses the correct stack sizes. We
// slightly change all the stack sizes in the configuration to catch problems
// with the used stack sizes not matching the configured sizes.

int pika_main()
{
    pika::this_thread::yield();

    pika::thread t([]() {});
    t.join();

    pika::finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
    pika::init_params p;
    p.cfg = {
#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__) ||            \
    defined(__APPLE__)
        "pika.stacks.small_size=" + std::to_string(PIKA_SMALL_STACK_SIZE + PIKA_EXEC_PAGESIZE),
        "pika.stacks.medium_size=" + std::to_string(PIKA_MEDIUM_STACK_SIZE + PIKA_EXEC_PAGESIZE),
        "pika.stacks.large_size=" + std::to_string(PIKA_LARGE_STACK_SIZE + PIKA_EXEC_PAGESIZE),
        "pika.stacks.huge_size=" + std::to_string(PIKA_HUGE_STACK_SIZE + PIKA_EXEC_PAGESIZE)
#endif
    };

    // This test should just run without crashing
    PIKA_TEST(true);
    return pika::init(pika_main, argc, argv, p);
}
