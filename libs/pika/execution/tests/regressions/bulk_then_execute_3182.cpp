//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// #3182: bulk_then_execute has unexpected return type/does not compile

#include <pika/local/algorithm.hpp>
#include <pika/local/execution.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <algorithm>
#include <atomic>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
std::atomic<int> void_count(0);
void fun1(int, pika::shared_future<int> f)
{
    PIKA_TEST(f.is_ready());
    PIKA_TEST_EQ(f.get(), 42);

    ++void_count;
}

std::atomic<int> int_count(0);
int fun2(int i, pika::shared_future<int> f)
{
    PIKA_TEST(f.is_ready());
    PIKA_TEST_EQ(f.get(), 42);

    ++int_count;
    return i;
}

template <typename Executor>
void test_bulk_then_execute(Executor&& exec)
{
    pika::shared_future<int> f = pika::make_ready_future(42);
    std::vector<int> v(100);
    std::iota(v.begin(), v.end(), 0);

    {
        pika::future<void> fut =
            pika::parallel::execution::bulk_then_execute(exec, &fun1, v, f);
        fut.get();

        PIKA_TEST_EQ(void_count.load(), 100);
    }

    {
        pika::future<std::vector<int>> fut =
            pika::parallel::execution::bulk_then_execute(exec, &fun2, v, f);
        auto result = fut.get();

        PIKA_TEST_EQ(int_count.load(), 100);
        PIKA_TEST(result == v);
    }
}

int pika_main()
{
    {
        void_count.store(0);
        int_count.store(0);

        pika::execution::parallel_executor exec;
        test_bulk_then_execute(exec);
    }

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    return pika::local::init(pika_main, argc, argv);
}
