//  Copyright (c) 2014-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/container_algorithms/mismatch.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
int seed = std::random_device{}();
std::mt19937 gen(seed);
std::uniform_int_distribution<> dis(0, 10006);

template <typename IteratorTag>
void test_mismatch1(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    iterator begin1 = iterator(std::begin(c1));
    iterator end1 = iterator(std::end(c1));

    {
        auto result =
            pika::ranges::mismatch(begin1, end1, std::begin(c2), std::end(c2));

        // verify values
        PIKA_TEST_EQ(std::size_t(std::distance(begin1, result.in1)), c1.size());
        PIKA_TEST_EQ(
            std::size_t(std::distance(std::begin(c2), result.in2)), c2.size());
    }

    {
        std::size_t changed_idx = dis(gen);    //-V104
        ++c1[changed_idx];

        auto result =
            pika::ranges::mismatch(begin1, end1, std::begin(c2), std::end(c2));

        // verify values
        PIKA_TEST_EQ(
            std::size_t(std::distance(begin1, result.in1)), changed_idx);
        PIKA_TEST_EQ(std::size_t(std::distance(std::begin(c2), result.in2)),
            changed_idx);
    }
}

template <typename ExPolicy, typename IteratorTag>
void test_mismatch1(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    iterator begin1 = iterator(std::begin(c1));
    iterator end1 = iterator(std::end(c1));

    {
        auto result = pika::ranges::mismatch(
            policy, begin1, end1, std::begin(c2), std::end(c2));

        // verify values
        PIKA_TEST_EQ(std::size_t(std::distance(begin1, result.in1)), c1.size());
        PIKA_TEST_EQ(
            std::size_t(std::distance(std::begin(c2), result.in2)), c2.size());
    }

    {
        std::size_t changed_idx = dis(gen);    //-V104
        ++c1[changed_idx];

        auto result = pika::ranges::mismatch(
            policy, begin1, end1, std::begin(c2), std::end(c2));

        // verify values
        PIKA_TEST_EQ(
            std::size_t(std::distance(begin1, result.in1)), changed_idx);
        PIKA_TEST_EQ(std::size_t(std::distance(std::begin(c2), result.in2)),
            changed_idx);
    }
}

template <typename ExPolicy, typename IteratorTag>
void test_mismatch1_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    iterator begin1 = iterator(std::begin(c1));
    iterator end1 = iterator(std::end(c1));

    {
        auto f = pika::ranges::mismatch(
            p, begin1, end1, std::begin(c2), std::end(c2));
        f.wait();

        // verify values
        auto result = f.get();
        PIKA_TEST_EQ(std::size_t(std::distance(begin1, result.in1)), c1.size());
        PIKA_TEST_EQ(
            std::size_t(std::distance(std::begin(c2), result.in2)), c2.size());
    }

    {
        std::size_t changed_idx = dis(gen);    //-V104
        ++c1[changed_idx];

        auto f = pika::ranges::mismatch(
            p, begin1, end1, std::begin(c2), std::end(c2));
        f.wait();

        // verify values
        auto result = f.get();
        PIKA_TEST_EQ(
            std::size_t(std::distance(begin1, result.in1)), changed_idx);
        PIKA_TEST_EQ(std::size_t(std::distance(std::begin(c2), result.in2)),
            changed_idx);
    }
}

template <typename IteratorTag>
void test_mismatch1()
{
    using namespace pika::execution;

    test_mismatch1(IteratorTag());

    test_mismatch1(seq, IteratorTag());
    test_mismatch1(par, IteratorTag());
    test_mismatch1(par_unseq, IteratorTag());

    test_mismatch1_async(seq(task), IteratorTag());
    test_mismatch1_async(par(task), IteratorTag());
}

void mismatch_test1()
{
    test_mismatch1<std::random_access_iterator_tag>();
    test_mismatch1<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_mismatch2(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    iterator begin1 = iterator(std::begin(c1));
    iterator end1 = iterator(std::end(c1));

    {
        auto result = pika::ranges::mismatch(begin1, end1, std::begin(c2),
            std::end(c2), std::equal_to<std::size_t>());

        // verify values
        PIKA_TEST_EQ(std::size_t(std::distance(begin1, result.in1)), c1.size());
        PIKA_TEST_EQ(
            std::size_t(std::distance(std::begin(c2), result.in2)), c2.size());
    }

    {
        std::size_t changed_idx = dis(gen);    //-V104
        ++c1[changed_idx];

        auto result = pika::ranges::mismatch(begin1, end1, std::begin(c2),
            std::end(c2), std::equal_to<std::size_t>());

        // verify values
        PIKA_TEST_EQ(
            std::size_t(std::distance(begin1, result.in1)), changed_idx);
        PIKA_TEST_EQ(std::size_t(std::distance(std::begin(c2), result.in2)),
            changed_idx);
    }
}

template <typename ExPolicy, typename IteratorTag>
void test_mismatch2(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    iterator begin1 = iterator(std::begin(c1));
    iterator end1 = iterator(std::end(c1));

    {
        auto result = pika::ranges::mismatch(policy, begin1, end1,
            std::begin(c2), std::end(c2), std::equal_to<std::size_t>());

        // verify values
        PIKA_TEST_EQ(std::size_t(std::distance(begin1, result.in1)), c1.size());
        PIKA_TEST_EQ(
            std::size_t(std::distance(std::begin(c2), result.in2)), c2.size());
    }

    {
        std::size_t changed_idx = dis(gen);    //-V104
        ++c1[changed_idx];

        auto result = pika::ranges::mismatch(policy, begin1, end1,
            std::begin(c2), std::end(c2), std::equal_to<std::size_t>());

        // verify values
        PIKA_TEST_EQ(
            std::size_t(std::distance(begin1, result.in1)), changed_idx);
        PIKA_TEST_EQ(std::size_t(std::distance(std::begin(c2), result.in2)),
            changed_idx);
    }
}

template <typename ExPolicy, typename IteratorTag>
void test_mismatch2_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    iterator begin1 = iterator(std::begin(c1));
    iterator end1 = iterator(std::end(c1));

    {
        auto f = pika::ranges::mismatch(p, begin1, end1, std::begin(c2),
            std::end(c2), std::equal_to<std::size_t>());
        f.wait();

        // verify values
        auto result = f.get();
        PIKA_TEST_EQ(std::size_t(std::distance(begin1, result.in1)), c1.size());
        PIKA_TEST_EQ(
            std::size_t(std::distance(std::begin(c2), result.in2)), c2.size());
    }

    {
        std::size_t changed_idx = dis(gen);    //-V104
        ++c1[changed_idx];

        auto f = pika::ranges::mismatch(p, begin1, end1, std::begin(c2),
            std::end(c2), std::equal_to<std::size_t>());
        f.wait();

        // verify values
        auto result = f.get();
        PIKA_TEST_EQ(
            std::size_t(std::distance(begin1, result.in1)), changed_idx);
        PIKA_TEST_EQ(std::size_t(std::distance(std::begin(c2), result.in2)),
            changed_idx);
    }
}

template <typename IteratorTag>
void test_mismatch2()
{
    using namespace pika::execution;

    test_mismatch2(IteratorTag());

    test_mismatch2(seq, IteratorTag());
    test_mismatch2(par, IteratorTag());
    test_mismatch2(par_unseq, IteratorTag());

    test_mismatch2_async(seq(task), IteratorTag());
    test_mismatch2_async(par(task), IteratorTag());
}

void mismatch_test2()
{
    test_mismatch2<std::random_access_iterator_tag>();
    test_mismatch2<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_mismatch_exception(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    bool caught_exception = false;
    try
    {
        pika::ranges::mismatch(iterator(std::begin(c1)), iterator(std::end(c1)),
            std::begin(c2), std::end(c2), [](std::size_t, std::size_t) {
                return throw std::runtime_error("test"), true;
            });

        PIKA_TEST(false);
    }
    catch (pika::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions<pika::execution::sequenced_policy,
            IteratorTag>::call(pika::execution::seq, e);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_mismatch_exception(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    bool caught_exception = false;
    try
    {
        pika::ranges::mismatch(policy, iterator(std::begin(c1)),
            iterator(std::end(c1)), std::begin(c2), std::end(c2),
            [](std::size_t, std::size_t) {
                return throw std::runtime_error("test"), true;
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
void test_mismatch_exception_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = pika::ranges::mismatch(p, iterator(std::begin(c1)),
            iterator(std::end(c1)), std::begin(c2), std::end(c2),
            [](std::size_t, std::size_t) {
                return throw std::runtime_error("test"), true;
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
void test_mismatch_exception()
{
    using namespace pika::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_mismatch_exception(seq, IteratorTag());
    test_mismatch_exception(par, IteratorTag());

    test_mismatch_exception_async(seq(task), IteratorTag());
    test_mismatch_exception_async(par(task), IteratorTag());
}

void mismatch_exception_test()
{
    test_mismatch_exception<std::random_access_iterator_tag>();
    test_mismatch_exception<std::forward_iterator_tag>();
}

/////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_mismatch_bad_alloc(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    bool caught_bad_alloc = false;
    try
    {
        pika::ranges::mismatch(policy, iterator(std::begin(c1)),
            iterator(std::end(c1)), std::begin(c2), std::end(c2),
            [](std::size_t, std::size_t) {
                return throw std::bad_alloc(), true;
            });

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
void test_mismatch_bad_alloc_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = pika::ranges::mismatch(p, iterator(std::begin(c1)),
            iterator(std::end(c1)), std::begin(c2), std::end(c2),
            [](std::size_t, std::size_t) {
                return throw std::bad_alloc(), true;
            });
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
void test_mismatch_bad_alloc()
{
    using namespace pika::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_mismatch_bad_alloc(seq, IteratorTag());
    test_mismatch_bad_alloc(par, IteratorTag());

    test_mismatch_bad_alloc_async(seq(task), IteratorTag());
    test_mismatch_bad_alloc_async(par(task), IteratorTag());
}

void mismatch_bad_alloc_test()
{
    test_mismatch_bad_alloc<std::random_access_iterator_tag>();
    test_mismatch_bad_alloc<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    mismatch_test1();
    mismatch_test2();
    mismatch_exception_test();
    mismatch_bad_alloc_test();
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
