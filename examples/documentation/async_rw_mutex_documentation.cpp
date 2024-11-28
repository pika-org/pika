//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/async_rw_mutex.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>

#include <fmt/printf.h>

#include <type_traits>
#include <utility>

int main(int argc, char* argv[])
{
    namespace ex = pika::execution::experimental;
    namespace tt = pika::this_thread::experimental;

    pika::start(argc, argv);
    ex::thread_pool_scheduler sched{};

    {
        ex::async_rw_mutex<int> m{0};

        // This read-write access is guaranteed to not run concurrently with any
        // other accesses. It will also run first since we requested the sender
        // first from the mutex.
        auto rw_access1 =
            m.readwrite() | ex::continues_on(sched) | ex::then([](auto w) {
                w.get() = 13;
                fmt::print("updated value to {}\n", w.get());
            });

        // These read-only accesses can only read the value, but they can run
        // concurrently. They'll see the write from the access above.
        auto ro_access1 =
            m.read() | ex::continues_on(sched) | ex::then([](auto w) {
                static_assert(std::is_const_v<
                    std::remove_reference_t<decltype(w.get())>>);
                fmt::print("value is now {}\n", w.get());
            });
        auto ro_access2 =
            m.read() | ex::continues_on(sched) | ex::then([](auto w) {
                static_assert(std::is_const_v<
                    std::remove_reference_t<decltype(w.get())>>);
                fmt::print("value is {} here as well\n", w.get());
            });
        auto ro_access3 =
            m.read() | ex::continues_on(sched) | ex::then([](auto w) {
                static_assert(std::is_const_v<
                    std::remove_reference_t<decltype(w.get())>>);
                fmt::print("and {} here too\n", w.get());
            });

        // This read-write access will run once all the above read-only accesses
        // are done.
        auto rw_access2 =
            m.readwrite() | ex::continues_on(sched) | ex::then([](auto w) {
                w.get() = 42;
                fmt::print("value is {} at the end\n", w.get());
            });

        // Start and wait for all the work to finish.
        tt::sync_wait(ex::when_all(std::move(rw_access1), std::move(ro_access1),
            std::move(ro_access2), std::move(ro_access3),
            std::move(rw_access2)));
    }

    pika::finalize();
    pika::stop();

    return 0;
}
