//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/algorithms/copy.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);
std::uniform_int_distribution<> dis(0, (std::numeric_limits<int>::max)());

void test_copy_if_seq()
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, std::forward_iterator_tag>
        iterator;

    std::vector<int> c(10007);
    std::vector<int> d(c.size());
    auto middle = std::begin(c) + c.size() / 2;
    std::iota(std::begin(c), middle, dis(gen));
    std::fill(middle, std::end(c), -1);

    pika::copy_if(iterator(std::begin(c)), iterator(std::end(c)), std::begin(d),
        [](int i) { return !(i < 0); });

    std::size_t count = 0;
    PIKA_TEST(std::equal(
        std::begin(c), middle, std::begin(d), [&count](int v1, int v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));

    PIKA_TEST(std::equal(middle, std::end(c), std::begin(d) + d.size() / 2,
        [&count](int v1, int v2) -> bool {
            PIKA_TEST_NEQ(v1, v2);
            ++count;
            return v1 != v2;
        }));

    PIKA_TEST_EQ(count, d.size());
}

template <typename ExPolicy>
void test_copy_if(ExPolicy&& policy)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, std::forward_iterator_tag>
        iterator;

    std::vector<int> c(10007);
    std::vector<int> d(c.size());
    auto middle = std::begin(c) + c.size() / 2;
    std::iota(std::begin(c), middle, dis(gen));
    std::fill(middle, std::end(c), -1);

    pika::copy_if(policy, iterator(std::begin(c)), iterator(std::end(c)),
        std::begin(d), [](int i) { return !(i < 0); });

    std::size_t count = 0;
    PIKA_TEST(std::equal(
        std::begin(c), middle, std::begin(d), [&count](int v1, int v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));

    PIKA_TEST(std::equal(middle, std::end(c), std::begin(d) + d.size() / 2,
        [&count](int v1, int v2) -> bool {
            PIKA_TEST_NEQ(v1, v2);
            ++count;
            return v1 != v2;
        }));

    PIKA_TEST_EQ(count, d.size());
}

template <typename ExPolicy>
void test_copy_if_async(ExPolicy&& p)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, std::forward_iterator_tag>
        iterator;

    std::vector<int> c(10007);
    std::vector<int> d(c.size());
    auto middle = std::begin(c) + c.size() / 2;
    std::iota(std::begin(c), middle, dis(gen));
    std::fill(middle, std::end(c), -1);

    auto f = pika::copy_if(p, iterator(std::begin(c)), iterator(std::end(c)),
        std::begin(d), [](int i) { return !(i < 0); });
    f.wait();

    std::size_t count = 0;
    PIKA_TEST(std::equal(
        std::begin(c), middle, std::begin(d), [&count](int v1, int v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));

    PIKA_TEST(std::equal(middle, std::end(c), std::begin(d) + d.size() / 2,
        [&count](int v1, int v2) -> bool {
            PIKA_TEST_NEQ(v1, v2);
            ++count;
            return v1 != v2;
        }));

    PIKA_TEST_EQ(count, d.size());
}

void test_copy_if()
{
    test_copy_if_seq();

    test_copy_if(pika::execution::seq);
    test_copy_if(pika::execution::par);
    test_copy_if(pika::execution::par_unseq);

    test_copy_if_async(pika::execution::seq(pika::execution::task));
    test_copy_if_async(pika::execution::par(pika::execution::task));
}

int pika_main(pika::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    test_copy_if();
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
