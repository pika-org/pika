//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/iterator_support/iterator_range.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/container_algorithms/uninitialized_copy.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include <pika/iterator_support/tests/iter_sent.hpp>
#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_uninitialized_copy_n_sent(IteratorTag)
{
    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), std::rand());
    std::copy(std::begin(c), std::end(c), std::rbegin(d));
    std::size_t sent_len = (std::rand() % 10007) + 1;
    pika::ranges::uninitialized_copy_n(std::begin(c), sent_len, std::begin(d),
        sentinel<std::size_t>{*(std::begin(d) + sent_len)});

    std::size_t count = 0;
    // loop till for sent_len since either the sentinel for the input or output iterator
    // will be reached by then
    PIKA_TEST(std::equal(std::begin(c), std::begin(c) + sent_len, std::begin(d),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));

    PIKA_TEST_EQ(count, sent_len);
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_copy_n_sent(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), std::rand());
    std::copy(std::begin(c), std::end(c), std::rbegin(d));
    std::size_t sent_len = (std::rand() % 10007) + 1;
    pika::ranges::uninitialized_copy_n(policy, std::begin(c), sent_len,
        std::begin(d), sentinel<std::size_t>{*(std::begin(d) + sent_len)});

    std::size_t count = 0;
    // loop till for sent_len since either the sentinel for the input or output iterator
    // will be reached by then
    PIKA_TEST(std::equal(std::begin(c), std::begin(c) + sent_len, std::begin(d),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));

    PIKA_TEST_EQ(count, sent_len);
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_copy_n_sent_async(ExPolicy&& p, IteratorTag)
{
    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), std::rand());
    std::copy(std::begin(c), std::end(c), std::rbegin(d));
    std::size_t sent_len = (std::rand() % 10007) + 1;
    auto f = pika::ranges::uninitialized_copy_n(p, std::begin(c), sent_len,
        std::begin(d), sentinel<std::size_t>{*(std::begin(d) + sent_len)});
    f.wait();

    std::size_t count = 0;
    PIKA_TEST(std::equal(std::begin(c), std::begin(c) + sent_len, std::begin(d),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    PIKA_TEST_EQ(count, sent_len);
}

template <typename IteratorTag>
void test_uninitialized_copy_n_sent()
{
    using namespace pika::execution;

    test_uninitialized_copy_n_sent(IteratorTag());

    test_uninitialized_copy_n_sent(seq, IteratorTag());
    test_uninitialized_copy_n_sent(par, IteratorTag());
    test_uninitialized_copy_n_sent(par_unseq, IteratorTag());

    test_uninitialized_copy_n_sent_async(seq(task), IteratorTag());
    test_uninitialized_copy_n_sent_async(par(task), IteratorTag());
}

void uninitialized_copy_n_sent_test()
{
    test_uninitialized_copy_n_sent<std::random_access_iterator_tag>();
    test_uninitialized_copy_n_sent<std::forward_iterator_tag>();
}

int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    uninitialized_copy_n_sent_test();
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
