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

template <typename ExPolicy, typename IteratorTag>
void test_for_loop_reduction_plus(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    std::size_t sum = 0;
    auto f =
        pika::for_loop(std::forward<ExPolicy>(policy), iterator(std::begin(c)),
            iterator(std::end(c)), pika::parallel::reduction_plus(sum),
            [](iterator it, std::size_t& sum) { sum += *it; });
    f.wait();

    // verify values
    std::size_t sum2 =
        std::accumulate(std::begin(c), std::end(c), std::size_t(0));
    PIKA_TEST_EQ(sum, sum2);
}

template <typename ExPolicy, typename IteratorTag>
void test_for_loop_reduction_multiplies(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    std::size_t prod = 0;
    auto f =
        pika::for_loop(std::forward<ExPolicy>(policy), iterator(std::begin(c)),
            iterator(std::end(c)), pika::parallel::reduction_multiplies(prod),
            [](iterator it, std::size_t& prod) { prod *= *it; });
    f.wait();

    // verify values
    std::size_t prod2 = std::accumulate(std::begin(c), std::end(c),
        std::size_t(1), std::multiplies<std::size_t>());
    PIKA_TEST_EQ(prod, prod2);
}

template <typename ExPolicy, typename IteratorTag>
void test_for_loop_reduction_min(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    std::size_t minval = c[0];

    auto f =
        pika::for_loop(std::forward<ExPolicy>(policy), iterator(std::begin(c)),
            iterator(std::end(c)), pika::parallel::reduction_min(minval),
            [](iterator it, std::size_t& minval) {
                minval = (std::min)(minval, *it);
            });
    f.wait();

    // verify values
    std::size_t minval2 = std::accumulate(std::begin(c), std::end(c), c[0],
        pika::parallel::v1::detail::min_of<std::size_t>());
    PIKA_TEST_EQ(minval, minval2);
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_for_loop_reduction()
{
    using namespace pika::execution;

    test_for_loop_reduction_plus(seq(task), IteratorTag());
    test_for_loop_reduction_plus(par(task), IteratorTag());

    test_for_loop_reduction_multiplies(seq(task), IteratorTag());
    test_for_loop_reduction_multiplies(par(task), IteratorTag());

    test_for_loop_reduction_min(seq(task), IteratorTag());
    test_for_loop_reduction_min(par(task), IteratorTag());
}

void for_loop_reduction_test()
{
    test_for_loop_reduction<std::random_access_iterator_tag>();
    test_for_loop_reduction<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_for_loop_reduction_bit_and_idx(ExPolicy&& policy)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    std::size_t bits = ~std::size_t(0);
    auto f = pika::for_loop(std::forward<ExPolicy>(policy), 0, c.size(),
        pika::parallel::reduction_bit_and(bits),
        [&c](std::size_t i, std::size_t& bits) { bits &= c[i]; });
    f.wait();

    // verify values
    std::size_t bits2 = std::accumulate(std::begin(c), std::end(c),
        ~std::size_t(0), std::bit_and<std::size_t>());
    PIKA_TEST_EQ(bits, bits2);
}

template <typename ExPolicy>
void test_for_loop_reduction_bit_or_idx(ExPolicy&& policy)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    std::size_t bits = 0;
    auto f = pika::for_loop(std::forward<ExPolicy>(policy), 0, c.size(),
        pika::parallel::reduction_bit_or(bits),
        [&c](std::size_t i, std::size_t& bits) { bits |= c[i]; });
    f.wait();

    // verify values
    std::size_t bits2 = std::accumulate(
        std::begin(c), std::end(c), std::size_t(0), std::bit_or<std::size_t>());
    PIKA_TEST_EQ(bits, bits2);
}

void for_loop_reduction_test_idx()
{
    using namespace pika::execution;

    test_for_loop_reduction_bit_and_idx(seq(task));
    test_for_loop_reduction_bit_and_idx(par(task));

    test_for_loop_reduction_bit_or_idx(seq(task));
    test_for_loop_reduction_bit_or_idx(par(task));
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    for_loop_reduction_test();
    for_loop_reduction_test_idx();

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
