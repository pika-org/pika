//  Copyright (c) 2018 Christopher Ogle
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/iterator_support/tests/iter_sent.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/container_algorithms/uninitialized_default_construct.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
struct default_constructable
{
    default_constructable()
      : value_(42)
    {
    }

    explicit default_constructable(std::int32_t val)
    {
        value_ = val;
    }

    bool operator!=(std::int32_t const& lhs) const
    {
        return lhs != value_;
    }

    std::int32_t value_;
};

struct value_constructable
{
    std::int32_t value_;
};

std::size_t const data_size = 10007;

////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_uninitialized_default_construct_range_sent(IteratorTag)
{
    typedef std::vector<default_constructable> base_iterator;

    base_iterator c(data_size, default_constructable(10));
    auto end_size = rand() % data_size;
    c[end_size] = default_constructable(20);

    pika::ranges::uninitialized_default_construct(
        std::begin(c), sentinel<std::int32_t>{20});

    std::size_t count42 = 0;
    std::size_t count10 = 0;
    std::for_each(std::begin(c), std::begin(c) + data_size,
        [&count42, &count10](default_constructable v1) {
            if (v1.value_ == 42)
            {
                count42++;
            }
            else if (v1.value_ == 10)
            {
                count10++;
            }
        });

    PIKA_TEST_EQ(count42, end_size);
    PIKA_TEST_EQ(count10, data_size - end_size - 1);
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_default_construct_range_sent(
    ExPolicy&& policy, IteratorTag)
{
    typedef std::vector<default_constructable> base_iterator;

    base_iterator c(data_size, default_constructable(10));
    auto end_size = rand() % data_size;
    c[end_size] = default_constructable(20);

    pika::ranges::uninitialized_default_construct(
        policy, std::begin(c), sentinel<std::int32_t>{20});

    std::size_t count42 = 0;
    std::size_t count10 = 0;
    std::for_each(std::begin(c), std::begin(c) + data_size,
        [&count42, &count10](default_constructable v1) {
            if (v1.value_ == 42)
            {
                count42++;
            }
            else if (v1.value_ == 10)
            {
                count10++;
            }
        });

    PIKA_TEST_EQ(count42, end_size);
    PIKA_TEST_EQ(count10, data_size - end_size - 1);
}

template <typename IteratorTag>
void test_uninitialized_default_construct_range(IteratorTag)
{
    typedef std::vector<default_constructable> base_iterator;

    base_iterator c(data_size, default_constructable(10));
    pika::ranges::uninitialized_default_construct(c);

    std::size_t count = 0;
    std::for_each(std::begin(c), std::begin(c) + data_size,
        [&count](default_constructable v1) {
            PIKA_TEST_EQ(v1.value_, 42);
            ++count;
        });
    PIKA_TEST_EQ(count, data_size);
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_default_construct_range(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<default_constructable> base_iterator;

    base_iterator c(data_size, default_constructable(10));
    pika::ranges::uninitialized_default_construct(
        std::forward<ExPolicy>(policy), c);

    std::size_t count = 0;
    std::for_each(std::begin(c), std::begin(c) + data_size,
        [&count](default_constructable v1) {
            PIKA_TEST_EQ(v1.value_, 42);
            ++count;
        });
    PIKA_TEST_EQ(count, data_size);
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_default_construct_range_async(
    ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<default_constructable> base_iterator;

    base_iterator c(data_size, default_constructable(10));
    auto f = pika::ranges::uninitialized_default_construct(
        std::forward<ExPolicy>(policy), c);
    f.wait();

    std::size_t count = 0;
    std::for_each(std::begin(c), std::begin(c) + data_size,
        [&count](default_constructable v1) {
            PIKA_TEST_EQ(v1.value_, 42);
            ++count;
        });
    PIKA_TEST_EQ(count, data_size);
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_default_construct_range2(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<value_constructable> base_iterator;
    base_iterator c(data_size, value_constructable{10});

    pika::ranges::uninitialized_default_construct(
        std::forward<ExPolicy>(policy), c);

    std::size_t count = 0;
    std::for_each(std::begin(c), std::begin(c) + data_size,
        [&count](value_constructable v1) {
            PIKA_TEST_EQ(v1.value_, (std::int32_t) 10);
            ++count;
        });
    PIKA_TEST_EQ(count, data_size);
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_default_construct_range_async2(
    ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<value_constructable> base_iterator;
    base_iterator c(data_size, value_constructable{10});

    auto f = pika::ranges::uninitialized_default_construct(
        std::forward<ExPolicy>(policy), c);
    f.wait();

    std::size_t count = 0;
    std::for_each(std::begin(c), std::begin(c) + data_size,
        [&count](value_constructable v1) {
            PIKA_TEST_EQ(v1.value_, (std::int32_t) 10);
            ++count;
        });
    PIKA_TEST_EQ(count, data_size);
}

template <typename IteratorTag>
void test_uninitialized_default_construct_range()
{
    using namespace pika::execution;

    test_uninitialized_default_construct_range(IteratorTag());
    test_uninitialized_default_construct_range(seq, IteratorTag());
    test_uninitialized_default_construct_range(par, IteratorTag());
    test_uninitialized_default_construct_range(par_unseq, IteratorTag());

    test_uninitialized_default_construct_range_async(seq(task), IteratorTag());
    test_uninitialized_default_construct_range_async(par(task), IteratorTag());

    test_uninitialized_default_construct_range2(seq, IteratorTag());
    test_uninitialized_default_construct_range2(par, IteratorTag());
    test_uninitialized_default_construct_range2(par_unseq, IteratorTag());

    test_uninitialized_default_construct_range_async2(seq(task), IteratorTag());
    test_uninitialized_default_construct_range_async2(par(task), IteratorTag());

    test_uninitialized_default_construct_range_sent(IteratorTag());
    test_uninitialized_default_construct_range_sent(seq, IteratorTag());
    test_uninitialized_default_construct_range_sent(par, IteratorTag());
    test_uninitialized_default_construct_range_sent(par_unseq, IteratorTag());
}

void uninitialized_default_construct_range_test()
{
    test_uninitialized_default_construct_range<
        std::random_access_iterator_tag>();
    test_uninitialized_default_construct_range<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    uninitialized_default_construct_range_test();
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
