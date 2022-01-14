//  Copyright (c) 2018 Christopher Ogle
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/iterator_support/tests/iter_sent.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/container_algorithms/ends_with.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_ends_with_sent(IteratorTag)
{
    using base_iterator = std::vector<int>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;
    using sentinel = test::sentinel_from_iterator<iterator>;

    auto end1 = std::rand() % 10007 + 1;
    auto end2 = std::rand() % end1;
    auto some_ints = std::vector<int>(end1);
    std::iota(some_ints.begin(), some_ints.end(), 1);

    auto some_more_ints =
        std::vector<int>(some_ints.begin() + end2, some_ints.end());

    auto some_wrong_ints = std::vector<int>(end1 - end2);
    std::iota(some_wrong_ints.begin(), some_wrong_ints.end(), -2);

    auto result1 = pika::ranges::ends_with(iterator(std::begin(some_ints)),
        sentinel(std::end(some_ints)), iterator(std::begin(some_more_ints)),
        sentinel(std::end(some_more_ints)));
    PIKA_TEST_EQ(result1, true);

    auto result2 = pika::ranges::ends_with(iterator(std::begin(some_ints)),
        sentinel(std::end(some_ints)), iterator(std::begin(some_wrong_ints)),
        sentinel(std::end(some_wrong_ints)));
    PIKA_TEST_EQ(result2, false);
}

template <typename IteratorTag, typename ExPolicy>
void test_ends_with_sent(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    using base_iterator = std::vector<int>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;
    using sentinel = test::sentinel_from_iterator<iterator>;

    auto end1 = std::rand() % 10007 + 1;
    auto end2 = std::rand() % end1;
    auto some_ints = std::vector<int>(end1);
    std::iota(some_ints.begin(), some_ints.end(), 1);

    auto some_more_ints =
        std::vector<int>(some_ints.begin() + end2, some_ints.end());

    auto some_wrong_ints = std::vector<int>(end1 - end2);
    std::iota(some_wrong_ints.begin(), some_wrong_ints.end(), -2);

    auto result1 =
        pika::ranges::ends_with(policy, iterator(std::begin(some_ints)),
            sentinel(std::end(some_ints)), iterator(std::begin(some_more_ints)),
            sentinel(std::end(some_more_ints)));
    PIKA_TEST_EQ(result1, true);

    auto result2 = pika::ranges::ends_with(policy,
        iterator(std::begin(some_ints)), sentinel(std::end(some_ints)),
        iterator(std::begin(some_wrong_ints)),
        sentinel(std::end(some_wrong_ints)));
    PIKA_TEST_EQ(result2, false);
}

template <typename IteratorTag>
void test_ends_with(IteratorTag)
{
    auto end1 = std::rand() % 10007 + 1;
    auto end2 = std::rand() % end1;
    auto some_ints = std::vector<int>(end1);
    std::iota(some_ints.begin(), some_ints.end(), 1);

    auto some_more_ints =
        std::vector<int>(some_ints.begin() + end2, some_ints.end());

    auto some_wrong_ints = std::vector<int>(end1 - end2);
    std::iota(some_wrong_ints.begin(), some_wrong_ints.end(), -2);

    auto result1 = pika::ranges::ends_with(some_ints, some_more_ints);
    PIKA_TEST_EQ(result1, true);

    auto result2 = pika::ranges::ends_with(some_ints, some_wrong_ints);
    PIKA_TEST_EQ(result2, false);
}

template <typename ExPolicy, typename IteratorTag>
void test_ends_with(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    auto end1 = std::rand() % 10007 + 1;
    auto end2 = std::rand() % end1;
    auto some_ints = std::vector<int>(end1);
    std::iota(some_ints.begin(), some_ints.end(), 1);

    auto some_more_ints =
        std::vector<int>(some_ints.begin() + end2, some_ints.end());

    auto some_wrong_ints = std::vector<int>(end1 - end2);
    std::iota(some_wrong_ints.begin(), some_wrong_ints.end(), -2);

    auto result1 = pika::ranges::ends_with(policy, some_ints, some_more_ints);
    PIKA_TEST_EQ(result1, true);

    auto result2 = pika::ranges::ends_with(policy, some_ints, some_wrong_ints);
    PIKA_TEST_EQ(result2, false);
}

template <typename ExPolicy, typename IteratorTag>
void test_ends_with_async(ExPolicy p, IteratorTag)
{
    auto end1 = std::rand() % 10007 + 1;
    auto end2 = std::rand() % end1;
    auto some_ints = std::vector<int>(end1);
    std::iota(some_ints.begin(), some_ints.end(), 1);

    auto some_more_ints =
        std::vector<int>(some_ints.begin() + end2, some_ints.end());

    auto some_wrong_ints = std::vector<int>(end1 - end2);
    std::iota(some_wrong_ints.begin(), some_wrong_ints.end(), -2);

    pika::future<bool> result1 =
        pika::ranges::ends_with(p, some_ints, some_more_ints);
    result1.wait();
    PIKA_TEST_EQ(result1.get(), true);

    pika::future<bool> result2 =
        pika::ranges::ends_with(p, some_ints, some_wrong_ints);
    result2.wait();
    PIKA_TEST_EQ(result2.get(), false);
}

template <typename IteratorTag>
void test_ends_with()
{
    using namespace pika::execution;

    test_ends_with_sent(IteratorTag());
    test_ends_with_sent(seq, IteratorTag());
    test_ends_with_sent(par, IteratorTag());
    test_ends_with_sent(par_unseq, IteratorTag());

    test_ends_with(IteratorTag());
    test_ends_with(seq, IteratorTag());
    test_ends_with(par, IteratorTag());
    test_ends_with(par_unseq, IteratorTag());

    test_ends_with_async(seq(task), IteratorTag());
    test_ends_with_async(par(task), IteratorTag());
}

void ends_with_test()
{
    test_ends_with<std::random_access_iterator_tag>();
    test_ends_with<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    ends_with_test();
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
