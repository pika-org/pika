//  Copyright (c) 2015 Daniel Bourgeois
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/functional/deferred_call.hpp>
#include <pika/iterator_support/iterator_range.hpp>
#include <pika/local/execution.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using pika::util::deferred_call;
typedef std::vector<int>::iterator iter;

////////////////////////////////////////////////////////////////////////////////
// A parallel executor that returns void for bulk_execute and pika::future<void>
// for bulk_async_execute
struct void_parallel_executor : pika::execution::parallel_executor
{
    using base_type = pika::execution::parallel_executor;

    template <typename F, typename Shape, typename... Ts>
    std::vector<pika::future<void>> bulk_async_execute(
        F&& f, Shape const& shape, Ts&&... ts)
    {
        std::vector<pika::future<void>> results;
        for (auto const& elem : shape)
        {
            results.push_back(this->base_type::async_execute(f, elem, ts...));
        }
        return results;
    }

    template <typename F, typename Shape, typename... Ts>
    void bulk_sync_execute(F&& f, Shape const& shape, Ts&&... ts)
    {
        return pika::unwrap(bulk_async_execute(
            std::forward<F>(f), shape, std::forward<Ts>(ts)...));
    }
};

namespace pika { namespace parallel { namespace execution {
    template <>
    struct is_two_way_executor<void_parallel_executor> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<void_parallel_executor> : std::true_type
    {
    };
}}}    // namespace pika::parallel::execution

////////////////////////////////////////////////////////////////////////////////
// Tests to void_parallel_executor behavior for the bulk executes

void bulk_test(int, pika::thread::id tid, int passed_through)    //-V813
{
    PIKA_TEST_NEQ(tid, pika::this_thread::get_id());
    PIKA_TEST_EQ(passed_through, 42);
}

void test_void_bulk_sync()
{
    typedef void_parallel_executor executor;

    pika::thread::id tid = pika::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    executor exec;
    pika::parallel::execution::bulk_sync_execute(
        exec, pika::util::bind(&bulk_test, _1, tid, _2), v, 42);
    pika::parallel::execution::bulk_sync_execute(exec, &bulk_test, v, tid, 42);
}

void test_void_bulk_async()
{
    typedef void_parallel_executor executor;

    pika::thread::id tid = pika::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    executor exec;
    pika::when_all(pika::parallel::execution::bulk_async_execute(
                      exec, pika::util::bind(&bulk_test, _1, tid, _2), v, 42))
        .get();
    pika::when_all(pika::parallel::execution::bulk_async_execute(
                      exec, &bulk_test, v, tid, 42))
        .get();
}

////////////////////////////////////////////////////////////////////////////////
// Sum using pika's parallel_executor and the above void_parallel_executor

// Create shape argument for parallel_executor
std::vector<pika::util::iterator_range<iter>> split(
    iter first, iter last, int parts)
{
    typedef std::iterator_traits<iter>::difference_type sz_type;
    sz_type count = std::distance(first, last);
    sz_type increment = count / parts;

    std::vector<pika::util::iterator_range<iter>> results;
    while (first != last)
    {
        iter prev = first;
        std::advance(first, (std::min)(increment, std::distance(first, last)));
        results.push_back(pika::util::make_iterator_range(prev, first));
    }
    return results;
}

// parallel sum using pika's parallel executor
int parallel_sum(iter first, iter last, int num_parts)
{
    pika::execution::parallel_executor exec;

    std::vector<pika::util::iterator_range<iter>> input =
        split(first, last, num_parts);

    std::vector<pika::future<int>> v =
        pika::parallel::execution::bulk_async_execute(
            exec,
            [](pika::util::iterator_range<iter> const& rng) -> int {
                return std::accumulate(std::begin(rng), std::end(rng), 0);
            },
            input);

    return std::accumulate(std::begin(v), std::end(v), 0,
        [](int a, pika::future<int>& b) -> int { return a + b.get(); });
}

// parallel sum using void parallel executer
int void_parallel_sum(iter first, iter last, int num_parts)
{
    void_parallel_executor exec;

    std::vector<int> temp(num_parts + 1, 0);
    std::iota(std::begin(temp), std::end(temp), 0);

    std::ptrdiff_t section_size = std::distance(first, last) / num_parts;

    std::vector<pika::future<void>> f =
        pika::parallel::execution::bulk_async_execute(
            exec,
            [&](const int& i) {
                iter b = first + i * section_size;    //-V104
                iter e = first +
                    (std::min)(std::distance(first, last),
                        static_cast<std::ptrdiff_t>(
                            (i + 1) * section_size)    //-V104
                    );
                temp[i] = std::accumulate(b, e, 0);    //-V108
            },
            temp);

    pika::when_all(f).get();

    return std::accumulate(std::begin(temp), std::end(temp), 0);
}

void sum_test()
{
    std::vector<int> vec(10007);
    auto random_num = []() { return std::rand() % 50 - 25; };
    std::generate(std::begin(vec), std::end(vec), random_num);

    int sum = std::accumulate(std::begin(vec), std::end(vec), 0);
    int num_parts = std::rand() % 5 + 3;

    // Return futures holding results of parallel_sum and void_parallel_sum
    pika::execution::parallel_executor exec;

    pika::future<int> f_par = pika::parallel::execution::async_execute(
        exec, &parallel_sum, std::begin(vec), std::end(vec), num_parts);

    pika::future<int> f_void_par = pika::parallel::execution::async_execute(
        exec, &void_parallel_sum, std::begin(vec), std::end(vec), num_parts);

    PIKA_TEST_EQ(f_par.get(), sum);
    PIKA_TEST_EQ(f_void_par.get(), sum);
}

////////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_void_bulk_sync();
    test_void_bulk_async();
    sum_test();
    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace pika::program_options;
    options_description desc_commandline(
        "Usage: " PIKA_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run");

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv, init_args), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
