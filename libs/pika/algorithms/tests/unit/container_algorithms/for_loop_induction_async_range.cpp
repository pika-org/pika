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

template <typename ExPolicy>
void test_for_loop_induction(ExPolicy&& policy)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    using iterator = std::vector<std::size_t>::iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(10007);
    std::iota(std::begin(c), std::end(c), gen());

    auto f = pika::ranges::for_loop(std::forward<ExPolicy>(policy), c,
        pika::parallel::induction(0), [&d](iterator it, std::size_t i) {
            *it = 42;
            d[i] = 42;
        });
    f.wait();

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](std::size_t v) -> void {
        PIKA_TEST_EQ(v, std::size_t(42));
        ++count;
    });
    std::for_each(std::begin(d), std::end(d),
        [](std::size_t v) -> void { PIKA_TEST_EQ(v, std::size_t(42)); });
    PIKA_TEST_EQ(count, c.size());
}

template <typename ExPolicy>
void test_for_loop_induction_stride(ExPolicy&& policy)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    using iterator = std::vector<std::size_t>::iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(10007);
    std::iota(std::begin(c), std::end(c), gen());

    auto f = pika::ranges::for_loop(std::forward<ExPolicy>(policy), c,
        pika::parallel::induction(0), pika::parallel::induction(0, 2),
        [&d](iterator it, std::size_t i, std::size_t j) {
            *it = 42;
            d[i] = 42;
            PIKA_TEST_EQ(2 * i, j);
        });
    f.wait();

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](std::size_t v) -> void {
        PIKA_TEST_EQ(v, std::size_t(42));
        ++count;
    });
    std::for_each(std::begin(d), std::end(d),
        [](std::size_t v) -> void { PIKA_TEST_EQ(v, std::size_t(42)); });
    PIKA_TEST_EQ(count, c.size());
}

template <typename ExPolicy>
void test_for_loop_induction_life_out(ExPolicy&& policy)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    using iterator = std::vector<std::size_t>::iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(10007);
    std::iota(std::begin(c), std::end(c), gen());

    std::size_t curr = 0;

    auto f = pika::ranges::for_loop(std::forward<ExPolicy>(policy), c,
        pika::parallel::induction(curr), [&d](iterator it, std::size_t i) {
            *it = 42;
            d[i] = 42;
        });
    f.wait();
    PIKA_TEST_EQ(curr, c.size());

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](std::size_t v) -> void {
        PIKA_TEST_EQ(v, std::size_t(42));
        ++count;
    });
    std::for_each(std::begin(d), std::end(d),
        [](std::size_t v) -> void { PIKA_TEST_EQ(v, std::size_t(42)); });
    PIKA_TEST_EQ(count, c.size());
}

template <typename ExPolicy>
void test_for_loop_induction_stride_life_out(ExPolicy&& policy)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    using iterator = std::vector<std::size_t>::iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(10007);
    std::iota(std::begin(c), std::end(c), gen());

    std::size_t curr1 = 0;
    std::size_t curr2 = 0;

    auto f = pika::ranges::for_loop(std::forward<ExPolicy>(policy), c,
        pika::parallel::induction(curr1), pika::parallel::induction(curr2, 2),
        [&d](iterator it, std::size_t i, std::size_t j) {
            *it = 42;
            d[i] = 42;
            PIKA_TEST_EQ(2 * i, j);
        });
    f.wait();
    PIKA_TEST_EQ(curr1, c.size());
    PIKA_TEST_EQ(curr2, 2 * c.size());

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](std::size_t v) -> void {
        PIKA_TEST_EQ(v, std::size_t(42));
        ++count;
    });
    std::for_each(std::begin(d), std::end(d),
        [](std::size_t v) -> void { PIKA_TEST_EQ(v, std::size_t(42)); });
    PIKA_TEST_EQ(count, c.size());
}

///////////////////////////////////////////////////////////////////////////////
void test_for_loop_induction()
{
    test_for_loop_induction(pika::execution::seq(pika::execution::task));
    test_for_loop_induction_stride(pika::execution::seq(pika::execution::task));
    test_for_loop_induction_life_out(pika::execution::par(pika::execution::task));
    test_for_loop_induction_stride_life_out(
        pika::execution::par(pika::execution::task));
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    test_for_loop_induction();

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
