//  Copyright (c) 2018 Christopher Ogle
//  Copyright (c) 2020 Hartmut Kaiser
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/iterator_support/tests/iter_sent.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/container_algorithms/inclusive_scan.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
unsigned int seed;
std::mt19937 gen;
std::uniform_int_distribution<> dis(1, 10006);

template <typename IteratorTag>
void test_inclusive_scan_sent(IteratorTag)
{
    auto end_len = dis(gen);
    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(end_len);
    std::vector<std::size_t> d1(end_len);
    std::vector<std::size_t> d2(end_len);
    std::fill(std::begin(c), std::begin(c) + end_len, std::size_t(1));
    c[end_len] = 2;

    std::size_t const val(0);
    auto op = [](std::size_t v1, std::size_t v2) { return v1 + v2; };

    auto res1 = pika::ranges::inclusive_scan(
        std::begin(c), sentinel<std::size_t>{2}, std::begin(d), op, val);

    auto res2 = pika::ranges::inclusive_scan(
        std::begin(c), sentinel<std::size_t>{2}, std::begin(d1), op);

    auto res3 = pika::ranges::inclusive_scan(
        std::begin(c), sentinel<std::size_t>{2}, std::begin(d2));

    PIKA_TEST(res1.in == std::begin(c) + end_len);
    PIKA_TEST(res1.out == std::begin(d) + end_len);

    PIKA_TEST(res2.in == std::begin(c) + end_len);
    PIKA_TEST(res2.out == std::begin(d1) + end_len);

    PIKA_TEST(res3.in == std::begin(c) + end_len);
    PIKA_TEST(res3.out == std::begin(d2) + end_len);

    // verify values
    std::vector<std::size_t> e(end_len);
    pika::parallel::v1::detail::sequential_inclusive_scan(
        std::begin(c), std::begin(c) + end_len, std::begin(e), val, op);

    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(e)));
    PIKA_TEST(std::equal(std::begin(d1), std::end(d1), std::begin(e)));
    PIKA_TEST(std::equal(std::begin(d2), std::end(d2), std::begin(e)));
}

template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan_sent(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    auto end_len = dis(gen);
    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(end_len);
    std::vector<std::size_t> d1(end_len);
    std::vector<std::size_t> d2(end_len);
    std::fill(std::begin(c), std::begin(c) + end_len, std::size_t(1));
    c[end_len] = 2;

    std::size_t const val(0);
    auto op = [](std::size_t v1, std::size_t v2) { return v1 + v2; };

    auto res1 = pika::ranges::inclusive_scan(policy, std::begin(c),
        sentinel<std::size_t>{2}, std::begin(d), op, val);

    auto res2 = pika::ranges::inclusive_scan(
        policy, std::begin(c), sentinel<std::size_t>{2}, std::begin(d1), op);

    auto res3 = pika::ranges::inclusive_scan(
        policy, std::begin(c), sentinel<std::size_t>{2}, std::begin(d2));

    PIKA_TEST(res1.in == std::begin(c) + end_len);
    PIKA_TEST(res1.out == std::begin(d) + end_len);

    PIKA_TEST(res2.in == std::begin(c) + end_len);
    PIKA_TEST(res2.out == std::begin(d1) + end_len);

    PIKA_TEST(res3.in == std::begin(c) + end_len);
    PIKA_TEST(res3.out == std::begin(d2) + end_len);

    // verify values
    std::vector<std::size_t> e(end_len);
    pika::parallel::v1::detail::sequential_inclusive_scan(
        std::begin(c), std::begin(c) + end_len, std::begin(e), val, op);

    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(e)));
    PIKA_TEST(std::equal(std::begin(d1), std::end(d1), std::begin(e)));
    PIKA_TEST(std::equal(std::begin(d2), std::end(d2), std::begin(e)));
}

template <typename IteratorTag>
void test_inclusive_scan(IteratorTag)
{
    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::vector<std::size_t> d1(c.size());
    std::vector<std::size_t> d2(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    std::size_t const val(0);
    auto op = [](std::size_t v1, std::size_t v2) { return v1 + v2; };

    auto res1 = pika::ranges::inclusive_scan(c, std::begin(d), op, val);
    auto res2 = pika::ranges::inclusive_scan(c, std::begin(d1), op);
    auto res3 = pika::ranges::inclusive_scan(c, std::begin(d2));

    PIKA_TEST(res1.in == std::end(c));
    PIKA_TEST(res1.out == std::end(d));

    PIKA_TEST(res2.in == std::end(c));
    PIKA_TEST(res2.out == std::end(d1));

    PIKA_TEST(res3.in == std::end(c));
    PIKA_TEST(res3.out == std::end(d2));

    // verify values
    std::vector<std::size_t> e(c.size());
    pika::parallel::v1::detail::sequential_inclusive_scan(
        std::begin(c), std::end(c), std::begin(e), val, op);

    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(e)));
    PIKA_TEST(std::equal(std::begin(d1), std::end(d1), std::begin(e)));
    PIKA_TEST(std::equal(std::begin(d2), std::end(d2), std::begin(e)));
}

template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::vector<std::size_t> d1(c.size());
    std::vector<std::size_t> d2(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    std::size_t const val(0);
    auto op = [](std::size_t v1, std::size_t v2) { return v1 + v2; };

    auto res1 = pika::ranges::inclusive_scan(policy, c, std::begin(d), op, val);
    auto res2 = pika::ranges::inclusive_scan(policy, c, std::begin(d1), op);
    auto res3 = pika::ranges::inclusive_scan(policy, c, std::begin(d2));

    PIKA_TEST(res1.in == std::end(c));
    PIKA_TEST(res1.out == std::end(d));

    PIKA_TEST(res2.in == std::end(c));
    PIKA_TEST(res2.out == std::end(d1));

    PIKA_TEST(res3.in == std::end(c));
    PIKA_TEST(res3.out == std::end(d2));

    // verify values
    std::vector<std::size_t> e(c.size());
    pika::parallel::v1::detail::sequential_inclusive_scan(
        std::begin(c), std::end(c), std::begin(e), val, op);

    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(e)));
    PIKA_TEST(std::equal(std::begin(d1), std::end(d1), std::begin(e)));
    PIKA_TEST(std::equal(std::begin(d2), std::end(d2), std::begin(e)));
}

template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan_async(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::vector<std::size_t> d1(c.size());
    std::vector<std::size_t> d2(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    std::size_t const val(0);
    auto op = [](std::size_t v1, std::size_t v2) { return v1 + v2; };

    pika::future<void> fut =
        pika::ranges::inclusive_scan(policy, c, std::begin(d), op, val);
    fut.wait();

    pika::future<void> fut1 =
        pika::ranges::inclusive_scan(policy, c, std::begin(d1), op);
    fut1.wait();

    pika::future<void> fut2 =
        pika::ranges::inclusive_scan(policy, c, std::begin(d2));
    fut2.wait();

    // verify values
    std::vector<std::size_t> e(c.size());
    pika::parallel::v1::detail::sequential_inclusive_scan(
        std::begin(c), std::end(c), std::begin(e), val, op);

    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(e)));
    PIKA_TEST(std::equal(std::begin(d1), std::end(d1), std::begin(e)));
    PIKA_TEST(std::equal(std::begin(d2), std::end(d2), std::begin(e)));
}

template <typename IteratorTag>
void test_inclusive_scan()
{
    using namespace pika::execution;

    test_inclusive_scan(IteratorTag());
    test_inclusive_scan(seq, IteratorTag());
    test_inclusive_scan(par, IteratorTag());
    test_inclusive_scan(par_unseq, IteratorTag());

    test_inclusive_scan_async(seq(task), IteratorTag());
    test_inclusive_scan_async(par(task), IteratorTag());

    test_inclusive_scan_sent(IteratorTag());
    test_inclusive_scan_sent(seq, IteratorTag());
    test_inclusive_scan_sent(par, IteratorTag());
    test_inclusive_scan_sent(par_unseq, IteratorTag());
}

void inclusive_scan_test()
{
    test_inclusive_scan<std::random_access_iterator_tag>();
    test_inclusive_scan<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed1 = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed1 = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed1 << std::endl;
    std::srand(seed1);

    seed = seed1;
    gen = std::mt19937(seed);

    inclusive_scan_test();
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
