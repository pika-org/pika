//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test reproduces a segfault in the split sender adaptor which happens if
// the continuations are cleared after all continuations are called. This may
// happen because the last continuation to run may reset the shared state of the
// split adaptor. If the shared state has been reset the continuations have
// already been released. There is a corresponding comment in the set_value
// implementation of split_receiver.

#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/testing.hpp>

#include <cstddef>
#include <cstdlib>

namespace ex = pika::execution::experimental;

int pika_main()
{
    // split cancellation is currently racy with stdexec:
    // https://github.com/NVIDIA/stdexec/issues/1426
#if !defined(PIKA_HAVE_STDEXEC)
    ex::thread_pool_scheduler sched;

    for (std::size_t i = 0; i < 100; ++i)
    {
        auto s = ex::schedule(sched) | ex::split();
        for (std::size_t j = 0; j < 10; ++j) { ex::start_detached(s); }
    }
#endif

    pika::finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::init(pika_main, argc, argv), 0, "pika main exited with non-zero status");

    return 0;
}
