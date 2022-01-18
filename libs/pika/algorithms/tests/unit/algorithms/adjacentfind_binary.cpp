//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/algorithm.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

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
std::uniform_int_distribution<> dis(2, 101);
std::uniform_int_distribution<> dist(2, 10005);

template <typename ExPolicy, typename IteratorTag>
void test_adjacent_find(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //fill vector with random values about 1
    std::iota(std::begin(c), std::end(c), dis(gen));

    std::size_t random_pos = dist(gen);    //-V101

    c[random_pos] = 100000;
    c[random_pos + 1] = 1;

    iterator index = pika::adjacent_find(policy, iterator(std::begin(c)),
        iterator(std::end(c)), std::greater<std::size_t>());

    base_iterator test_index = std::begin(c) + random_pos;

    PIKA_TEST(index == iterator(test_index));
}

template <typename ExPolicy, typename IteratorTag>
void test_adjacent_find_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    // fill vector with random values above 1
    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), dis(gen));

    std::size_t random_pos = dist(gen);    //-V101

    c[random_pos] = 100000;
    c[random_pos + 1] = 1;

    pika::future<iterator> f = pika::adjacent_find(p, iterator(std::begin(c)),
        iterator(std::end(c)), std::greater<std::size_t>());
    f.wait();

    // create iterator at position of value to be found
    base_iterator test_index = std::begin(c) + random_pos;
    PIKA_TEST(f.get() == iterator(test_index));
}

template <typename IteratorTag>
void test_adjacent_find()
{
    using namespace pika::execution;
    test_adjacent_find(seq, IteratorTag());
    test_adjacent_find(par, IteratorTag());
    test_adjacent_find(par_unseq, IteratorTag());

    test_adjacent_find_async(seq(task), IteratorTag());
    test_adjacent_find_async(par(task), IteratorTag());
}

void adjacent_find_test()
{
    test_adjacent_find<std::random_access_iterator_tag>();
    test_adjacent_find<std::forward_iterator_tag>();
}

int pika_main(pika::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    adjacent_find_test();
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
