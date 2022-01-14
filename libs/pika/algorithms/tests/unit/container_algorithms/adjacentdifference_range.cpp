//  Copyright (c) 2015 Daniel Bourgeois
//  Copyright (c) 2021 Karame M.Shokooh
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/iterator_support/tests/iter_sent.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/container_algorithms/adjacent_difference.hpp>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_sen(ExPolicy policy)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c(10007);
    std::iota(c.begin(), c.end(), 1);
    std::vector<std::size_t> d(10007);
    std::vector<std::size_t> d_ans(10007);

    auto it = pika::ranges::adjacent_difference(
        policy, std::begin(c), sentinel<std::size_t>{10007}, std::begin(d));
    std::adjacent_difference(std::begin(c), std::end(c), std::begin(d_ans));

    PIKA_TEST(std::equal(std::begin(d), std::end(d) - 1, std::begin(d_ans),
        [](std::size_t lhs, std::size_t rhs) -> bool { return lhs == rhs; }));

    PIKA_TEST(std::end(d) - 1 == it);
}
void test_sen()
{
    std::vector<std::size_t> c(10007);
    std::iota(c.begin(), c.end(), 1);
    std::vector<std::size_t> d(10007);
    std::vector<std::size_t> d_ans(10007);

    auto it = pika::ranges::adjacent_difference(
        std::begin(c), sentinel<std::size_t>{10007}, std::begin(d));
    std::adjacent_difference(std::begin(c), std::end(c), std::begin(d_ans));

    PIKA_TEST(std::equal(std::begin(d), std::end(d) - 1, std::begin(d_ans),
        [](std::size_t lhs, std::size_t rhs) -> bool { return lhs == rhs; }));

    PIKA_TEST(std::end(d) - 1 == it);
}

template <typename ExPolicy>
void test_adjacent_difference(ExPolicy policy)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c = test::random_iota(10007);
    std::vector<std::size_t> d(10007);
    std::vector<std::size_t> d_ans(10007);

    auto it = pika::ranges::adjacent_difference(policy, c, std::begin(d));
    std::adjacent_difference(std::begin(c), std::end(c), std::begin(d_ans));

    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(d_ans),
        [](std::size_t lhs, std::size_t rhs) -> bool { return lhs == rhs; }));

    PIKA_TEST(std::end(d) == it);
}
void test_adjacent_difference()
{
    std::vector<std::size_t> c = test::random_iota(10007);
    std::vector<std::size_t> d(10007);
    std::vector<std::size_t> d_ans(10007);

    auto it = pika::ranges::adjacent_difference(c, std::begin(d));
    std::adjacent_difference(std::begin(c), std::end(c), std::begin(d_ans));

    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(d_ans),
        [](std::size_t lhs, std::size_t rhs) -> bool { return lhs == rhs; }));

    PIKA_TEST(std::end(d) == it);
}

template <typename ExPolicy>
void test_adjacent_difference_async(ExPolicy p)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c = test::random_iota(10007);
    std::vector<std::size_t> d(10007);
    std::vector<std::size_t> d_ans(10007);

    auto f_it = pika::ranges::adjacent_difference(p, c, std::begin(d));
    std::adjacent_difference(std::begin(c), std::end(c), std::begin(d_ans));

    f_it.wait();
    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(d_ans),
        [](std::size_t lhs, std::size_t rhs) -> bool { return lhs == rhs; }));

    PIKA_TEST(std::end(d) == f_it.get());
}

void adjacent_difference_test()
{
    using namespace pika::execution;
    test_adjacent_difference(seq);
    test_adjacent_difference(par);
    test_adjacent_difference(par_unseq);
    test_sen(seq);
    test_sen(par);
    test_sen(par_unseq);

    test_adjacent_difference_async(seq(task));
    test_adjacent_difference_async(par(task));
    test_adjacent_difference();
    test_sen();
}

int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    adjacent_difference_test();
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
