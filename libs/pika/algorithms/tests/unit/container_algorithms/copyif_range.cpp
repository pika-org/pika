//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/container_algorithms/copy.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
void test_copy_if_seq()
{
    std::vector<int> c(10007);
    std::vector<int> d(c.size());
    auto middle = std::begin(c) + c.size() / 2;
    std::iota(std::begin(c), middle, std::rand());
    std::fill(middle, std::end(c), -1);

    pika::ranges::copy_if(c, std::begin(d), [](int i) { return !(i < 0); });

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

    std::vector<int> c(10007);
    std::vector<int> d(c.size());
    auto middle = std::begin(c) + c.size() / 2;
    std::iota(std::begin(c), middle, std::rand());
    std::fill(middle, std::end(c), -1);

    pika::ranges::copy_if(
        policy, c, std::begin(d), [](int i) { return !(i < 0); });

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
    std::vector<int> c(10007);
    std::vector<int> d(c.size());
    auto middle = std::begin(c) + c.size() / 2;
    std::iota(std::begin(c), middle, std::rand());
    std::fill(middle, std::end(c), -1);

    auto f = pika::ranges::copy_if(
        p, c, std::begin(d), [](int i) { return !(i < 0); });
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
    using namespace pika::execution;

    test_copy_if_seq();

    test_copy_if(seq);
    test_copy_if(par);
    test_copy_if(par_unseq);

    test_copy_if_async(seq(task));
    test_copy_if_async(par(task));
}

int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

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
