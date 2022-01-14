//  Copyright (c) 2017-2018 Taeguk Kwon
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/iterator_support/tests/iter_sent.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/container_algorithms/unique.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
struct user_defined_type
{
    user_defined_type() = default;
    user_defined_type(int rand_no)
      : val(rand_no)
      , name(name_list[std::rand() % name_list.size()])
    {
    }

    bool operator==(user_defined_type const& t) const
    {
        return this->name == t.name && this->val == t.val;
    }

    bool operator!=(user_defined_type const& t) const
    {
        return this->name != t.name || this->val != t.val;
    }

    static const std::vector<std::string> name_list;

    int val;
    std::string name;
};

const std::vector<std::string> user_defined_type::name_list{
    "ABB", "ABC", "ACB", "BASE", "CAA", "CAAA", "CAAB"};

struct random_fill
{
    random_fill(int rand_base, int range)
      : gen(std::rand())
      , dist(rand_base - range / 2, rand_base + range / 2)
    {
    }

    int operator()()
    {
        return dist(gen);
    }

    std::mt19937 gen;
    std::uniform_int_distribution<> dist;
};

////////////////////////////////////////////////////////////////////////////
void test_unique_sent()
{
    std::size_t const size = 10007;
    std::vector<std::size_t> c(size), d;
    std::generate(std::begin(c), std::end(c),
        []() -> std::size_t { return std::rand() % 10; });
    d = c;

    auto end_len = std::rand() % 10006 + 1;
    c[end_len] = 10;

    auto result = pika::ranges::unique(std::begin(c), sentinel<std::size_t>{10});
    auto solution = std::unique(std::begin(d), std::begin(d) + end_len);

    bool equality =
        test::equal(std::begin(c), result.begin(), std::begin(d), solution);

    PIKA_TEST(equality);
}

template <typename ExPolicy>
void test_unique_sent(ExPolicy policy)
{
    std::size_t const size = 10007;
    std::vector<std::size_t> c(size), d;
    std::generate(std::begin(c), std::end(c),
        []() -> std::size_t { return std::rand() % 10; });
    d = c;

    auto end_len = std::rand() % 10006 + 1;
    c[end_len] = 10;

    auto result =
        pika::ranges::unique(policy, std::begin(c), sentinel<std::size_t>{10});
    auto solution = std::unique(std::begin(d), std::begin(d) + end_len);

    bool equality =
        test::equal(std::begin(c), result.begin(), std::begin(d), solution);

    PIKA_TEST(equality);
}

////////////////////////////////////////////////////////////////////////////
template <typename DataType>
void test_unique(DataType)
{
    std::size_t const size = 10007;
    std::vector<DataType> c(size), d;
    std::generate(std::begin(c), std::end(c), random_fill(0, 6));
    d = c;

    auto result = pika::ranges::unique(c);
    auto solution = std::unique(std::begin(d), std::end(d));

    bool equality =
        test::equal(std::begin(c), result.begin(), std::begin(d), solution);

    PIKA_TEST(equality);
}

template <typename ExPolicy, typename DataType>
void test_unique(ExPolicy policy, DataType)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    using pika::get;

    std::size_t const size = 10007;
    std::vector<DataType> c(size), d;
    std::generate(std::begin(c), std::end(c), random_fill(0, 6));
    d = c;

    auto result = pika::ranges::unique(policy, c);
    auto solution = std::unique(std::begin(d), std::end(d));

    bool equality =
        test::equal(std::begin(c), result.begin(), std::begin(d), solution);

    PIKA_TEST(equality);
}

template <typename ExPolicy, typename DataType>
void test_unique_async(ExPolicy policy, DataType)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    using pika::get;

    std::size_t const size = 10007;
    std::vector<DataType> c(size), d;
    std::generate(std::begin(c), std::end(c), random_fill(0, 6));
    d = c;

    auto f = pika::ranges::unique(policy, c);
    auto result = f.get();
    auto solution = std::unique(std::begin(d), std::end(d));

    bool equality =
        test::equal(std::begin(c), result.begin(), std::begin(d), solution);

    PIKA_TEST(equality);
}

template <typename DataType>
void test_unique()
{
    using namespace pika::execution;

    test_unique_sent();
    test_unique_sent(seq);
    test_unique_sent(par);
    test_unique_sent(par_unseq);

    test_unique(DataType());
    test_unique(seq, DataType());
    test_unique(par, DataType());
    test_unique(par_unseq, DataType());

    test_unique_async(seq(task), DataType());
    test_unique_async(par(task), DataType());
}

void test_unique()
{
    test_unique<int>();
    test_unique<user_defined_type>();
}

int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_unique();
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
