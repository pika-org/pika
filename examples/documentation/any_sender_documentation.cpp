//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/init.hpp>

#include <fmt/printf.h>

#include <chrono>
#include <cstddef>
#include <string_view>
#include <thread>
#include <utility>

void print_answer(std::string_view message,
    pika::execution::experimental::unique_any_sender<int>&& sender)
{
    auto const answer =
        pika::this_thread::experimental::sync_wait(std::move(sender));
    fmt::print("{}: {}\n", message, answer);
}

int main(int argc, char* argv[])
{
    namespace ex = pika::execution::experimental;
    namespace tt = pika::this_thread::experimental;

    pika::start(argc, argv);
    ex::thread_pool_scheduler sched{};

    ex::unique_any_sender<int> sender;

    // Whether the sender is a simple just-sender...
    sender = ex::just(42);
    print_answer("Quick answer", std::move(sender));

    // ... or a more complicated sender, we can put them both into the same
    // unique_any_sender as long as they send the same types.
    sender = ex::schedule(sched) | ex::then([]() {
        std::this_thread::sleep_for(std::chrono::seconds(3));
        return 42;
    });
    print_answer("Slow answer", std::move(sender));

    // If we try to use the sender again it will throw an exception
    try
    {
        // NOLINTNEXTLINE(bugprone-use-after-move)
        tt::sync_wait(std::move(sender));
    }
    catch (std::exception const& e)
    {
        fmt::print("Caught exception: {}\n", e.what());
    }

    // We can also use a type-erased sender to chain work. The type of the
    // sender remains the same each iteration thanks to the type-erasure, but
    // the work it represents grows.
    //
    // However, note that using a specialized algorithm like repeat_n from
    // stdexec may be more efficient.
    ex::unique_any_sender<int> chain{ex::just(0)};
    for (std::size_t i = 0; i < 42; ++i)
    {
        chain = std::move(chain) | ex::continues_on(sched) |
            ex::then([](int x) { return x + 1; });
    }
    print_answer("Final answer", std::move(chain));

    pika::finalize();
    pika::stop();

    return 0;
}
