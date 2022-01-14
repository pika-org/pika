//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/local/condition_variable.hpp>
#include <pika/local/execution.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/mutex.hpp>
#include <pika/modules/testing.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>

///////////////////////////////////////////////////////////////////////////////
std::atomic<std::int32_t> accumulator;
pika::lcos::local::condition_variable_any result_cv;

void increment(std::int32_t i)
{
    accumulator += i;
    result_cv.notify_one();
}

void increment_with_future(pika::shared_future<std::int32_t> fi)
{
    accumulator += fi.get();
    result_cv.notify_one();
}

///////////////////////////////////////////////////////////////////////////////
struct increment_function_object
{
    void operator()(std::int32_t i) const
    {
        accumulator += i;
    }
};

///////////////////////////////////////////////////////////////////////////////
struct increment_type
{
    void call(std::int32_t i) const
    {
        accumulator += i;
    }
};

auto increment_lambda = [](std::int32_t i) { accumulator += i; };

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_apply_with_executor(Executor& exec)
{
    accumulator.store(0);

    {
        using pika::util::placeholders::_1;

        pika::apply(exec, &increment, 1);
        pika::apply(exec, pika::util::bind(&increment, 1));
        pika::apply(exec, pika::util::bind(&increment, _1), 1);
    }

    {
        pika::lcos::local::promise<std::int32_t> p;
        pika::shared_future<std::int32_t> f = p.get_future();

        p.set_value(1);

        using pika::util::placeholders::_1;

        pika::apply(exec, &increment_with_future, f);
        pika::apply(exec, pika::util::bind(&increment_with_future, f));
        pika::apply(exec, pika::util::bind(&increment_with_future, _1), f);
    }

    {
        using pika::util::placeholders::_1;

        pika::apply(exec, increment, 1);
        pika::apply(exec, pika::util::bind(increment, 1));
        pika::apply(exec, pika::util::bind(increment, _1), 1);
    }

    {
        increment_type inc;

        using pika::util::placeholders::_1;

        pika::apply(exec, &increment_type::call, inc, 1);
        pika::apply(exec, pika::util::bind(&increment_type::call, inc, 1));
        pika::apply(exec, pika::util::bind(&increment_type::call, inc, _1), 1);
    }

    {
        increment_function_object obj;

        using pika::util::placeholders::_1;

        pika::apply(exec, obj, 1);
        pika::apply(exec, pika::util::bind(obj, 1));
        pika::apply(exec, pika::util::bind(obj, _1), 1);
    }

    {
        using pika::util::placeholders::_1;

        pika::apply(exec, increment_lambda, 1);
        pika::apply(exec, pika::util::bind(increment_lambda, 1));
        pika::apply(exec, pika::util::bind(increment_lambda, _1), 1);
    }

    pika::lcos::local::no_mutex result_mutex;
    std::unique_lock<pika::lcos::local::no_mutex> l(result_mutex);
    result_cv.wait_for(l, std::chrono::seconds(1),
        pika::util::bind(
            std::equal_to<std::int32_t>(), std::ref(accumulator), 18));

    PIKA_TEST_EQ(accumulator.load(), 18);
}

int pika_main()
{
    {
        pika::execution::sequenced_executor exec;
        test_apply_with_executor(exec);
    }

    {
        pika::execution::parallel_executor exec;
        test_apply_with_executor(exec);
    }

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run pika
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
