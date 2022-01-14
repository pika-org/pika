//  Copyright (c) 2015 Daniel Bourgeois
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/functional/deferred_call.hpp>
#include <pika/local/execution.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
int bulk_test(
    pika::thread::id tid, int value, bool is_par, int passed_through)    //-V813
{
    PIKA_TEST_EQ(is_par, (tid != pika::this_thread::get_id()));
    PIKA_TEST_EQ(passed_through, 42);
    return value;
}

template <typename Executor>
void test_bulk_sync(Executor& exec)
{
    pika::thread::id tid = pika::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), 0);

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    std::vector<int> results = pika::parallel::execution::bulk_sync_execute(
        exec, pika::util::bind(&bulk_test, tid, _1, false, _2), v, 42);

    PIKA_TEST(std::equal(std::begin(results), std::end(results), std::begin(v)));
}

template <typename Executor>
void test_bulk_async(Executor& exec)
{
    pika::thread::id tid = pika::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), 0);

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    std::vector<pika::future<int>> results =
        pika::parallel::execution::bulk_async_execute(
            exec, pika::util::bind(&bulk_test, tid, _1, true, _2), v, 42);

    PIKA_TEST(std::equal(std::begin(results), std::end(results), std::begin(v),
        [](pika::future<int>& lhs, const int& rhs) {
            return lhs.get() == rhs;
        }));
}

////////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    pika::execution::sequenced_executor seq_exec;
    test_bulk_sync(seq_exec);

    pika::execution::parallel_executor par_exec;
    pika::execution::parallel_executor par_fork_exec(pika::launch::fork);
    test_bulk_async(par_exec);
    test_bulk_async(par_fork_exec);

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.cfg = cfg;

    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv, init_args), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
