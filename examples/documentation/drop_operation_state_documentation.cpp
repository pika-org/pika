//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/init.hpp>

#include <cassert>
#include <memory>
#include <utility>

int main(int argc, char* argv[])
{
    namespace ex = pika::execution::experimental;
    namespace tt = pika::this_thread::experimental;

    pika::start(argc, argv);
    ex::thread_pool_scheduler sched{};

    auto sp = std::make_shared<int>(42);
    std::weak_ptr<int> sp_weak = sp;

    auto s = ex::just(std::move(sp)) |
        ex::then([&](auto&&) { assert(sp_weak.use_count() == 1); }) |
        // Even though the shared_ptr is no longer in use, it may be kept alive
        // by the operation state
        ex::then([&]() {
            assert(sp_weak.use_count() == 1);
            return 42;
        }) |
        ex::drop_operation_state() |
        // Once drop_operation_state has been used, the shared_ptr is guaranteed
        // to be released.  Values are passed through the adaptor.
        ex::then([&]([[maybe_unused]] int x) {
            assert(sp_weak.use_count() == 0);
            assert(x == 42);
        });
    tt::sync_wait(std::move(s));

    pika::finalize();
    pika::stop();

    return 0;
}
