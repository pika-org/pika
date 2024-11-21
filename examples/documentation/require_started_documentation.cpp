//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/init.hpp>

#include <fmt/printf.h>

#include <cassert>
#include <cstdlib>
#include <exception>
#include <utility>

int main(int argc, char* argv[])
{
    namespace ex = pika::execution::experimental;
    namespace tt = pika::this_thread::experimental;

    pika::start(argc, argv);
    ex::thread_pool_scheduler sched{};

    {
        // require_started forwards values received from the predecessor sender
        auto s = ex::just(42) | ex::require_started() |
            ex::then([]([[maybe_unused]] auto&& i) { assert(i == 42); });
        tt::sync_wait(std::move(s));
    }

    {
        // The termination is ignored with discard, the sender is from the
        // user's perspective rightfully not used
        auto s = ex::just() | ex::require_started();
        s.discard();
    }

    {
        // The require_started sender terminates on destruction if it has not
        // been used
        auto s = ex::just() | ex::require_started();
    }
    assert(false);

    pika::finalize();
    pika::stop();
    return 0;
}
