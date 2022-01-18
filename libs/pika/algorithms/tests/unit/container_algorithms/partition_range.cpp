//  Copyright (c) 2017-2018 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/iterator_support/tests/iter_sent.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/container_algorithms/partition.hpp>
#include <pika/type_support/unused.hpp>

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

    bool operator<(int rand_base) const
    {
        static std::string const base_name = "BASE";

        if (this->name < base_name)
            return true;
        else if (this->name > base_name)
            return false;
        else
            return this->val < rand_base;
    }

    bool operator<(user_defined_type const& t) const
    {
        return this->name < t.name ||
            (this->name == t.name && this->val < t.val);
    }

    bool operator==(user_defined_type const& t) const
    {
        return this->name == t.name && this->val == t.val;
    }

    bool operator!=(user_defined_type const& t) const
    {
        return this->name != t.name || this->val != t.val;
    }

    struct user_defined_type& operator++()
    {
        return *this;
    };

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
void test_partition_sent()
{
    using pika::get;
    using DataType = int;

    int rand_base = std::rand();
    auto pred = [rand_base](
                    DataType const& t) -> bool { return t < rand_base; };

    std::size_t const size = 300007;
    std::vector<DataType> c(size), c_org;
    std::generate(
        std::begin(c), std::end(c), random_fill(rand_base, size / 10));
    c_org = c;
    c[size - 1] = INT_MAX;

    auto result = pika::ranges::partition(
        std::begin(c), sentinel<DataType>{INT_MAX}, pred);

    bool is_partitioned =
        std::is_partitioned(std::begin(c), std::end(c) - 1, pred);

    PIKA_TEST(is_partitioned);

    auto solution = std::partition_point(std::begin(c), std::end(c) - 1, pred);

    PIKA_TEST(result.begin() == solution);

    std::sort(std::begin(c), std::end(c) - 1);
    std::sort(std::begin(c_org), std::end(c_org) - 1);

    bool unchanged = test::equal(
        std::begin(c), std::end(c) - 1, std::begin(c_org), std::end(c_org) - 1);

    PIKA_TEST(unchanged);
}

template <typename ExPolicy>
void test_partition_sent(ExPolicy policy)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    using pika::get;
    using DataType = int;

    int rand_base = std::rand();
    auto pred = [rand_base](
                    DataType const& t) -> bool { return t < rand_base; };

    std::size_t const size = 300007;
    std::vector<DataType> c(size), c_org;
    std::generate(
        std::begin(c), std::end(c), random_fill(rand_base, size / 10));
    c_org = c;
    c[size - 1] = INT_MAX;

    auto result = pika::ranges::partition(
        policy, std::begin(c), sentinel<DataType>{INT_MAX}, pred);

    bool is_partitioned =
        std::is_partitioned(std::begin(c), std::end(c) - 1, pred);

    PIKA_TEST(is_partitioned);

    auto solution = std::partition_point(std::begin(c), std::end(c) - 1, pred);

    PIKA_TEST(result.begin() == solution);

    std::sort(std::begin(c), std::end(c) - 1);
    std::sort(std::begin(c_org), std::end(c_org) - 1);

    bool unchanged = test::equal(
        std::begin(c), std::end(c) - 1, std::begin(c_org), std::end(c_org) - 1);

    PIKA_TEST(unchanged);
}

template <typename DataType>
void test_partition(DataType)
{
    using pika::get;

    int rand_base = std::rand();
    auto pred = [rand_base](
                    DataType const& t) -> bool { return t < rand_base; };

    std::size_t const size = 300007;
    std::vector<DataType> c(size), c_org;
    std::generate(
        std::begin(c), std::end(c), random_fill(rand_base, size / 10));
    c_org = c;

    auto result = pika::ranges::partition(c, pred);

    bool is_partitioned = std::is_partitioned(std::begin(c), std::end(c), pred);

    PIKA_TEST(is_partitioned);

    auto solution = std::partition_point(std::begin(c), std::end(c), pred);

    PIKA_TEST(result.begin() == solution);

    std::sort(std::begin(c), std::end(c));
    std::sort(std::begin(c_org), std::end(c_org));

    bool unchanged = test::equal(
        std::begin(c), std::end(c), std::begin(c_org), std::end(c_org));

    PIKA_TEST(unchanged);
}

template <typename ExPolicy, typename DataType>
void test_partition(ExPolicy policy, DataType)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    using pika::get;

    int rand_base = std::rand();
    auto pred = [rand_base](
                    DataType const& t) -> bool { return t < rand_base; };

    std::size_t const size = 300007;
    std::vector<DataType> c(size), c_org;
    std::generate(
        std::begin(c), std::end(c), random_fill(rand_base, size / 10));
    c_org = c;

    auto result = pika::ranges::partition(policy, c, pred);

    bool is_partitioned = std::is_partitioned(std::begin(c), std::end(c), pred);

    PIKA_TEST(is_partitioned);

    auto solution = std::partition_point(std::begin(c), std::end(c), pred);

    PIKA_TEST(result.begin() == solution);

    std::sort(std::begin(c), std::end(c));
    std::sort(std::begin(c_org), std::end(c_org));

    bool unchanged = test::equal(
        std::begin(c), std::end(c), std::begin(c_org), std::end(c_org));

    PIKA_TEST(unchanged);
}

template <typename ExPolicy, typename DataType>
void test_partition_async(ExPolicy policy, DataType)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    using pika::get;

    int rand_base = std::rand();
    auto pred = [rand_base](
                    DataType const& t) -> bool { return t < rand_base; };

    std::size_t const size = 300007;
    std::vector<DataType> c(size), c_org;
    std::generate(
        std::begin(c), std::end(c), random_fill(rand_base, size / 10));
    c_org = c;

    auto f = pika::ranges::partition(policy, c, pred);
    auto result = f.get();

    bool is_partitioned = std::is_partitioned(std::begin(c), std::end(c), pred);

    PIKA_TEST(is_partitioned);

    auto solution = std::partition_point(std::begin(c), std::end(c), pred);

    PIKA_TEST(result.begin() == solution);

    std::sort(std::begin(c), std::end(c));
    std::sort(std::begin(c_org), std::end(c_org));

    bool unchanged = test::equal(
        std::begin(c), std::end(c), std::begin(c_org), std::end(c_org));

    PIKA_TEST(unchanged);
}

template <typename DataType>
void test_partition()
{
    using namespace pika::execution;

    test_partition(DataType());
    test_partition(seq, DataType());
    test_partition(par, DataType());
    test_partition(par_unseq, DataType());

    test_partition_async(seq(task), DataType());
    test_partition_async(par(task), DataType());

    test_partition_sent();
    test_partition_sent(seq);
    test_partition_sent(par);
    test_partition_sent(par_unseq);
}

void test_partition()
{
    test_partition<int>();
    test_partition<user_defined_type>();
}

int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_partition();
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
