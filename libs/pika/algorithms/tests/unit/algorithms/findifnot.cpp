//  copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/algorithms/find.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);
std::uniform_int_distribution<> dis(2, 101);

template <typename IteratorTag>
void test_find_if_not(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //fill vector with random values about 1
    std::fill(std::begin(c), std::end(c), dis(gen));
    c.at(c.size() / 2) = 1;

    iterator index =
        pika::find_if_not(iterator(std::begin(c)), iterator(std::end(c)),
            [](std::size_t v) { return v != std::size_t(1); });

    base_iterator test_index = std::begin(c) + c.size() / 2;

    PIKA_TEST(index == iterator(test_index));
}

template <typename ExPolicy, typename IteratorTag>
void test_find_if_not(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //fill vector with random values about 1
    std::fill(std::begin(c), std::end(c), dis(gen));
    c.at(c.size() / 2) = 1;

    iterator index =
        pika::find_if_not(policy, iterator(std::begin(c)), iterator(std::end(c)),
            [](std::size_t v) { return v != std::size_t(1); });

    base_iterator test_index = std::begin(c) + c.size() / 2;

    PIKA_TEST(index == iterator(test_index));
}

template <typename ExPolicy, typename IteratorTag>
void test_find_if_not_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //fill vector with random values above 1
    std::fill(std::begin(c), std::end(c), dis(gen));
    c.at(c.size() / 2) = 1;

    pika::future<iterator> f =
        pika::find_if_not(p, iterator(std::begin(c)), iterator(std::end(c)),
            [](std::size_t v) { return v != std::size_t(1); });
    f.wait();

    //create iterator at position of value to be found
    base_iterator test_index = std::begin(c) + c.size() / 2;

    PIKA_TEST(f.get() == iterator(test_index));
}

template <typename IteratorTag>
void test_find_if_not()
{
    using namespace pika::execution;

    test_find_if_not(IteratorTag());

    test_find_if_not(seq, IteratorTag());
    test_find_if_not(par, IteratorTag());
    test_find_if_not(par_unseq, IteratorTag());

    test_find_if_not_async(seq(task), IteratorTag());
    test_find_if_not_async(par(task), IteratorTag());
}

void find_if_not_test()
{
    test_find_if_not<std::random_access_iterator_tag>();
    test_find_if_not<std::forward_iterator_tag>();
}

int pika_main(pika::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    find_if_not_test();
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
