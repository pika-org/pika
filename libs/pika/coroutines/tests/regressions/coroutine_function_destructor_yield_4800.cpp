//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test verifies that the destructor of a thread function may yield.

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>
#include <pika/modules/threading.hpp>

#include <utility>

struct thread_function_yield_destructor
{
    pika::threads::thread_result_type operator()(pika::threads::thread_arg_type)
    {
        return {pika::threads::thread_schedule_state::terminated,
            pika::threads::invalid_thread_id};
    }

    ~thread_function_yield_destructor()
    {
        pika::this_thread::yield();
    }
};

struct yielder
{
    ~yielder()
    {
        pika::this_thread::yield();
    }
};

int pika_main()
{
    // We supply the thread function ourselves which means that the destructor
    // will be called late in the coroutine call operator.
    {
        pika::threads::thread_init_data data{thread_function_yield_destructor{},
            "thread_function_yield_destructor"};
        pika::threads::register_thread(data);
    }

    // This is a more complicated example which sometimes leads to the yielder
    // destructor being called late in the coroutine call operator.
    for (int i = 0; i < 1000; ++i)
    {
        pika::lcos::local::promise<yielder> p;
        pika::future<yielder> f = p.get_future();
        pika::dataflow([](auto&&) {}, f);
        p.set_value(yielder{});
    }

    // In the following two cases the yielder instance gets destructed earlier
    // in the coroutine call operator (before the thread function returns), so
    // these cases should never fail, even when the above two cases may fail.
    for (int i = 0; i < 1000; ++i)
    {
        pika::lcos::local::promise<yielder> p;
        pika::future<yielder> f = p.get_future();
        f.then([](auto&&) {});
        p.set_value(yielder{});
    }

    for (int i = 0; i < 1000; ++i)
    {
        yielder y;
        pika::apply([y = std::move(y)]() {});
    }

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
