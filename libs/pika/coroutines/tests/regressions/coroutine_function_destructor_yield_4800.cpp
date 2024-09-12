//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test verifies that the destructor of a thread function may yield.

#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/modules/threading.hpp>
#include <pika/testing.hpp>
#include <pika/thread.hpp>

#include <cstdlib>
#include <utility>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

struct thread_function_yield_destructor
{
    pika::threads::detail::thread_result_type operator()(pika::threads::detail::thread_arg_type)
    {
        return {pika::threads::detail::thread_schedule_state::terminated,
            pika::threads::detail::invalid_thread_id};
    }

    ~thread_function_yield_destructor() { pika::this_thread::yield(); }
};

struct yielder
{
    ~yielder() { pika::this_thread::yield(); }
};

int pika_main()
{
    // We supply the thread function ourselves which means that the destructor
    // will be called late in the coroutine call operator.
    {
        pika::threads::detail::thread_init_data data{
            thread_function_yield_destructor{}, "thread_function_yield_destructor"};
        pika::threads::detail::register_thread(data);
    }

#if defined(PIKA_WITH_VALGRIND)
    const int num_iterations = 100;
#else
    const int num_iterations = 1000;
#endif

    // This is a more complicated example which sometimes leads to the yielder
    // destructor being called late in the coroutine call operator.
    for (int i = 0; i < num_iterations; ++i)
    {
        tt::sync_wait(ex::just(yielder{}) | ex::continues_on(ex::thread_pool_scheduler{}) |
            ex::then([](auto&&) {}));
    }

    for (int i = 0; i < num_iterations; ++i)
    {
        ex::execute(ex::thread_pool_scheduler{}, [y = yielder{}] {});
    }

    pika::finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::init(pika_main, argc, argv), 0, "pika main exited with non-zero status");

    return 0;
}
