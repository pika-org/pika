//  Copyright (c) 2015 Daniel Bourgeois
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/algorithms/remove_copy.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
int seed = std::random_device{}();
std::mt19937 gen(seed);

template <typename IteratorTag>
void test_remove_copy_if(IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::vector<int> d(c.size());
    std::uniform_int_distribution<> dis(0, (c.size() >> 1) - 1);
    std::uniform_int_distribution<> dist(0, c.size() - 1);

    std::size_t middle_idx = dis(gen);
    auto middle = std::begin(c) + middle_idx;
    std::iota(std::begin(c), middle, static_cast<int>(dist(gen)));
    std::fill(middle, std::end(c), -1);

    pika::remove_copy_if(iterator(std::begin(c)), iterator(std::end(c)),
        std::begin(d), [](int i) { return i < 0; });

    std::size_t count = 0;
    PIKA_TEST(std::equal(
        std::begin(c), middle, std::begin(d), [&count](int v1, int v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));

    PIKA_TEST(std::equal(middle, std::end(c), std::begin(d) + middle_idx,
        [&count](int v1, int v2) -> bool {
            PIKA_TEST_NEQ(v1, v2);
            ++count;
            return v1 != v2;
        }));

    PIKA_TEST_EQ(count, d.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_remove_copy_if(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::vector<int> d(c.size());
    std::uniform_int_distribution<> dis(0, (c.size() >> 1) - 1);
    std::uniform_int_distribution<> dist(0, c.size() - 1);

    std::size_t middle_idx = dis(gen);
    auto middle = std::begin(c) + middle_idx;
    std::iota(std::begin(c), middle, static_cast<int>(dist(gen)));
    std::fill(middle, std::end(c), -1);

    pika::remove_copy_if(policy, iterator(std::begin(c)), iterator(std::end(c)),
        std::begin(d), [](int i) { return i < 0; });

    std::size_t count = 0;
    PIKA_TEST(std::equal(
        std::begin(c), middle, std::begin(d), [&count](int v1, int v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));

    PIKA_TEST(std::equal(middle, std::end(c), std::begin(d) + middle_idx,
        [&count](int v1, int v2) -> bool {
            PIKA_TEST_NEQ(v1, v2);
            ++count;
            return v1 != v2;
        }));

    PIKA_TEST_EQ(count, d.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_remove_copy_if_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::vector<int> d(c.size());
    std::uniform_int_distribution<> dis(0, (c.size() >> 1) - 1);
    std::uniform_int_distribution<> dist(0, c.size() - 1);

    std::size_t middle_idx = dis(gen);
    auto middle = std::begin(c) + middle_idx;
    std::iota(std::begin(c), middle, static_cast<int>(dist(gen)));
    std::fill(middle, std::end(c), -1);

    auto f = pika::remove_copy_if(p, iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(d), [](int i) { return i < 0; });
    f.wait();

    std::size_t count = 0;
    PIKA_TEST(std::equal(
        std::begin(c), middle, std::begin(d), [&count](int v1, int v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));

    PIKA_TEST(std::equal(middle, std::end(c), std::begin(d) + middle_idx,
        [&count](int v1, int v2) -> bool {
            PIKA_TEST_NEQ(v1, v2);
            ++count;
            return v1 != v2;
        }));

    PIKA_TEST_EQ(count, d.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_remove_copy_if_outiter(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::vector<int> d(0);
    std::uniform_int_distribution<> dis(0, (c.size() >> 1) - 1);
    std::uniform_int_distribution<> dist(0, c.size() - 1);

    std::size_t middle_idx = dis(gen);
    auto middle = std::begin(c) + middle_idx;
    std::iota(std::begin(c), middle, static_cast<int>(dist(gen)));
    std::fill(middle, std::end(c), -1);

    pika::remove_copy_if(policy, iterator(std::begin(c)), iterator(std::end(c)),
        std::back_inserter(d), [](int i) { return i < 0; });

    PIKA_TEST(std::equal(
        std::begin(c), middle, std::begin(d), [](int v1, int v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            return v1 == v2;
        }));

    PIKA_TEST_EQ(middle_idx, d.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_remove_copy_if_outiter_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::vector<int> d(0);
    std::uniform_int_distribution<> dis(0, (c.size() >> 1) - 1);
    std::uniform_int_distribution<> dist(0, c.size() - 1);

    std::size_t middle_idx = dis(gen);
    auto middle = std::begin(c) + middle_idx;
    std::iota(std::begin(c), middle, static_cast<int>(dist(gen)));
    std::fill(middle, std::end(c), -1);

    auto f =
        pika::remove_copy_if(p, iterator(std::begin(c)), iterator(std::end(c)),
            std::back_inserter(d), [](int i) { return i < 0; });
    f.wait();

    PIKA_TEST(std::equal(
        std::begin(c), middle, std::begin(d), [](int v1, int v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            return v1 == v2;
        }));

    PIKA_TEST_EQ(middle_idx, d.size());
}

template <typename IteratorTag>
void test_remove_copy_if()
{
    using namespace pika::execution;

    test_remove_copy_if(IteratorTag());
    test_remove_copy_if(seq, IteratorTag());
    test_remove_copy_if(par, IteratorTag());
    test_remove_copy_if(par_unseq, IteratorTag());

    test_remove_copy_if_async(seq(task), IteratorTag());
    test_remove_copy_if_async(par(task), IteratorTag());
}

void remove_copy_if_test()
{
    test_remove_copy_if<std::random_access_iterator_tag>();
    test_remove_copy_if<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_remove_copy_if_exception(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), gen());

    bool caught_exception = false;
    try
    {
        pika::remove_copy_if(policy, iterator(std::begin(c)),
            iterator(std::end(c)), std::begin(d), [](std::size_t v) {
                return throw std::runtime_error("test"), v == 0;
            });
        PIKA_TEST(false);
    }
    catch (pika::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_remove_copy_if_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), gen());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = pika::remove_copy_if(p, iterator(std::begin(c)),
            iterator(std::end(c)), std::begin(d), [](std::size_t v) {
                return throw std::runtime_error("test"), v == 0;
            });

        returned_from_algorithm = true;
        f.get();

        PIKA_TEST(false);
    }
    catch (pika::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(p, e);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);
    PIKA_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_remove_copy_if_exception()
{
    using namespace pika::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_remove_copy_if_exception(seq, IteratorTag());
    test_remove_copy_if_exception(par, IteratorTag());

    test_remove_copy_if_exception_async(seq(task), IteratorTag());
    test_remove_copy_if_exception_async(par(task), IteratorTag());
}

void remove_copy_if_exception_test()
{
    test_remove_copy_if_exception<std::random_access_iterator_tag>();
    test_remove_copy_if_exception<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_remove_copy_if_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), gen());

    bool caught_bad_alloc = false;
    try
    {
        pika::remove_copy_if(policy, iterator(std::begin(c)),
            iterator(std::end(c)), std::begin(d),
            [](std::size_t v) { return throw std::bad_alloc(), v; });

        PIKA_TEST(false);
    }
    catch (std::bad_alloc const&)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_bad_alloc);
}

template <typename ExPolicy, typename IteratorTag>
void test_remove_copy_if_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), gen());

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = pika::remove_copy_if(p, iterator(std::begin(c)),
            iterator(std::end(c)), std::begin(d),
            [](std::size_t v) { return throw std::bad_alloc(), v; });

        returned_from_algorithm = true;
        f.get();

        PIKA_TEST(false);
    }
    catch (std::bad_alloc const&)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_bad_alloc);
    PIKA_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_remove_copy_if_bad_alloc()
{
    using namespace pika::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_remove_copy_if_bad_alloc(seq, IteratorTag());
    test_remove_copy_if_bad_alloc(par, IteratorTag());

    test_remove_copy_if_bad_alloc_async(seq(task), IteratorTag());
    test_remove_copy_if_bad_alloc_async(par(task), IteratorTag());
}

void remove_copy_if_bad_alloc_test()
{
    test_remove_copy_if_bad_alloc<std::random_access_iterator_tag>();
    test_remove_copy_if_bad_alloc<std::forward_iterator_tag>();
}

int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    remove_copy_if_test();
    remove_copy_if_exception_test();
    remove_copy_if_bad_alloc_test();
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
