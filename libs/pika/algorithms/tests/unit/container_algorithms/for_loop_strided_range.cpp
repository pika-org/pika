//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/algorithm.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
int seed = std::random_device{}();
std::mt19937 gen(seed);
std::uniform_int_distribution<> dis(1, 10006);

template <typename ExPolicy>
void test_for_loop_strided(ExPolicy&& policy)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    using iterator = std::vector<std::size_t>::iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    std::for_each(std::begin(c), std::end(c), [](std::size_t& v) -> void {
        if (v == 42)
            v = 43;
    });

    int stride = dis(gen);    //-V103

    pika::ranges::for_loop_strided(std::forward<ExPolicy>(policy), c, stride,
        [](iterator it) { *it = 42; });

    // verify values
    std::size_t count = 0;
    for (std::size_t i = 0; i != c.size(); ++i)
    {
        if (i % stride == 0)    //-V104
        {
            PIKA_TEST_EQ(c[i], std::size_t(42));
        }
        else
        {
            PIKA_TEST_NEQ(c[i], std::size_t(42));
        }
        ++count;
    }
    PIKA_TEST_EQ(count, c.size());
}

template <typename ExPolicy>
void test_for_loop_strided_async(ExPolicy&& p)
{
    using iterator = std::vector<std::size_t>::iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    std::for_each(std::begin(c), std::end(c), [](std::size_t& v) -> void {
        if (v == 42)
            v = 43;
    });

    int stride = dis(gen);    //-V103

    auto f = pika::ranges::for_loop_strided(
        std::forward<ExPolicy>(p), c, stride, [](iterator it) { *it = 42; });
    f.wait();

    // verify values
    std::size_t count = 0;
    for (std::size_t i = 0; i != c.size(); ++i)
    {
        if (i % stride == 0)    //-V104
        {
            PIKA_TEST_EQ(c[i], std::size_t(42));
        }
        else
        {
            PIKA_TEST_NEQ(c[i], std::size_t(42));
        }
        ++count;
    }
    PIKA_TEST_EQ(count, c.size());
}

void test_for_loop_strided()
{
    test_for_loop_strided(pika::execution::seq);
    test_for_loop_strided(pika::execution::par);
    test_for_loop_strided(pika::execution::par_unseq);

    test_for_loop_strided_async(pika::execution::seq(pika::execution::task));
    test_for_loop_strided_async(pika::execution::par(pika::execution::task));
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    test_for_loop_strided();

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
