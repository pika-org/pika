//  Copyright (c) 2014-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/execution.hpp>
#include <pika/modules/iterator_support.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/algorithms/inclusive_scan.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
void inclusive_scan_benchmark()
{
    try
    {
#if defined(PIKA_DEBUG)
        std::vector<double> c(1000000);
#else
        std::vector<double> c(100000000);
#endif
        std::vector<double> d(c.size());
        std::fill(std::begin(c), std::end(c), 1.0);

        double const val(0);
        auto op = [](double v1, double v2) { return v1 + v2; };

        pika::chrono::high_resolution_timer t;
        pika::inclusive_scan(pika::execution::par, std::begin(c), std::end(c),
            std::begin(d), op, val);
        double elapsed = t.elapsed();

        // verify values
        std::vector<double> e(c.size());
        pika::parallel::v1::detail::sequential_inclusive_scan(
            std::begin(c), std::end(c), std::begin(e), val, op);

        bool ok = std::equal(std::begin(d), std::end(d), std::begin(e));
        PIKA_TEST(ok);
        if (ok)
        {
            // CDash graph plotting
            pika::util::print_cdash_timing("InclusiveScanTime", elapsed);
        }
    }
    catch (...)
    {
        PIKA_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_inclusive_scan1(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    std::size_t const val(0);
    auto op = [](std::size_t v1, std::size_t v2) { return v1 + v2; };

    pika::inclusive_scan(
        iterator(std::begin(c)), iterator(std::end(c)), std::begin(d), op, val);

    // verify values
    std::vector<std::size_t> e(c.size());
    pika::parallel::v1::detail::sequential_inclusive_scan(
        std::begin(c), std::end(c), std::begin(e), val, op);

    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(e)));

#if defined(PIKA_HAVE_CXX17_STD_SCAN_ALGORITHMS)
    std::vector<std::size_t> f(c.size());
    std::inclusive_scan(std::begin(c), std::end(c), std::begin(f), op, val);
    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(f)));
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan1(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    std::size_t const val(0);
    auto op = [](std::size_t v1, std::size_t v2) { return v1 + v2; };

    pika::inclusive_scan(std::forward<ExPolicy>(policy), iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(d), op, val);

    // verify values
    std::vector<std::size_t> e(c.size());
    pika::parallel::v1::detail::sequential_inclusive_scan(
        std::begin(c), std::end(c), std::begin(e), val, op);

    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(e)));

#if defined(PIKA_HAVE_CXX17_STD_SCAN_ALGORITHMS)
    std::vector<std::size_t> f(c.size());
    std::inclusive_scan(std::begin(c), std::end(c), std::begin(f), op, val);
    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(f)));
#endif
}

template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan1_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    std::size_t const val(0);
    auto op = [](std::size_t v1, std::size_t v2) { return v1 + v2; };

    pika::future<void> fut = pika::inclusive_scan(std::forward<ExPolicy>(p),
        iterator(std::begin(c)), iterator(std::end(c)), std::begin(d), op, val);
    fut.wait();

    // verify values
    std::vector<std::size_t> e(c.size());
    pika::parallel::v1::detail::sequential_inclusive_scan(
        std::begin(c), std::end(c), std::begin(e), val, op);

    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(e)));

#if defined(PIKA_HAVE_CXX17_STD_SCAN_ALGORITHMS)
    std::vector<std::size_t> f(c.size());
    std::inclusive_scan(std::begin(c), std::end(c), std::begin(f), op, val);
    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(f)));
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_inclusive_scan2(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    auto op = [](std::size_t v1, std::size_t v2) { return v1 + v2; };

    pika::inclusive_scan(
        iterator(std::begin(c)), iterator(std::end(c)), std::begin(d), op);

    // verify values
    std::vector<std::size_t> e(c.size());
    pika::parallel::v1::detail::sequential_inclusive_scan_noinit(
        std::begin(c), std::end(c), std::begin(e), op);

    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(e)));

#if defined(PIKA_HAVE_CXX17_STD_SCAN_ALGORITHMS)
    std::vector<std::size_t> f(c.size());
    std::inclusive_scan(std::begin(c), std::end(c), std::begin(f), op);
    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(f)));
#endif
}

template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan2(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    auto op = [](std::size_t v1, std::size_t v2) { return v1 + v2; };

    pika::inclusive_scan(policy, iterator(std::begin(c)), iterator(std::end(c)),
        std::begin(d), op);

    // verify values
    std::vector<std::size_t> e(c.size());
    pika::parallel::v1::detail::sequential_inclusive_scan_noinit(
        std::begin(c), std::end(c), std::begin(e), op);

    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(e)));

#if defined(PIKA_HAVE_CXX17_STD_SCAN_ALGORITHMS)
    std::vector<std::size_t> f(c.size());
    std::inclusive_scan(std::begin(c), std::end(c), std::begin(f), op);
    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(f)));
#endif
}

template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan2_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    auto op = [](std::size_t v1, std::size_t v2) { return v1 + v2; };

    pika::future<void> fut = pika::inclusive_scan(
        p, iterator(std::begin(c)), iterator(std::end(c)), std::begin(d), op);
    fut.wait();

    // verify values
    std::vector<std::size_t> e(c.size());
    pika::parallel::v1::detail::sequential_inclusive_scan_noinit(
        std::begin(c), std::end(c), std::begin(e), op);

    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(e)));

#if defined(PIKA_HAVE_CXX17_STD_SCAN_ALGORITHMS)
    std::vector<std::size_t> f(c.size());
    std::inclusive_scan(std::begin(c), std::end(c), std::begin(f), op);
    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(f)));
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_inclusive_scan3(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    pika::inclusive_scan(
        iterator(std::begin(c)), iterator(std::end(c)), std::begin(d));

    // verify values
    std::vector<std::size_t> e(c.size());
    pika::parallel::v1::detail::sequential_inclusive_scan_noinit(
        std::begin(c), std::end(c), std::begin(e), std::plus<std::size_t>());

    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(e)));

#if defined(PIKA_HAVE_CXX17_STD_SCAN_ALGORITHMS)
    std::vector<std::size_t> f(c.size());
    std::inclusive_scan(std::begin(c), std::end(c), std::begin(f));
    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(f)));
#endif
}

template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan3(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    pika::inclusive_scan(
        policy, iterator(std::begin(c)), iterator(std::end(c)), std::begin(d));

    // verify values
    std::vector<std::size_t> e(c.size());
    pika::parallel::v1::detail::sequential_inclusive_scan_noinit(
        std::begin(c), std::end(c), std::begin(e), std::plus<std::size_t>());

    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(e)));

#if defined(PIKA_HAVE_CXX17_STD_SCAN_ALGORITHMS)
    std::vector<std::size_t> f(c.size());
    std::inclusive_scan(std::begin(c), std::end(c), std::begin(f));
    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(f)));
#endif
}

template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan3_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    pika::future<void> fut = pika::inclusive_scan(
        p, iterator(std::begin(c)), iterator(std::end(c)), std::begin(d));
    fut.wait();

    // verify values
    std::vector<std::size_t> e(c.size());
    pika::parallel::v1::detail::sequential_inclusive_scan_noinit(
        std::begin(c), std::end(c), std::begin(e), std::plus<std::size_t>());

    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(e)));

#if defined(PIKA_HAVE_CXX17_STD_SCAN_ALGORITHMS)
    std::vector<std::size_t> f(c.size());
    std::inclusive_scan(std::begin(c), std::end(c), std::begin(f));
    PIKA_TEST(std::equal(std::begin(d), std::end(d), std::begin(f)));
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan_exception(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    bool caught_exception = false;
    try
    {
        pika::inclusive_scan(
            policy, iterator(std::begin(c)), iterator(std::end(c)),
            std::begin(d),
            [](std::size_t v1, std::size_t v2) {
                return throw std::runtime_error("test"), v1 + v2;
            },
            std::size_t(0));

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
void test_inclusive_scan_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        pika::future<void> f = pika::inclusive_scan(
            p, iterator(std::begin(c)), iterator(std::end(c)), std::begin(d),
            [](std::size_t v1, std::size_t v2) {
                return throw std::runtime_error("test"), v1 + v2;
            },
            std::size_t(0));

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

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    bool caught_exception = false;
    try
    {
        pika::inclusive_scan(
            policy, iterator(std::begin(c)), iterator(std::end(c)),
            std::begin(d),
            [](std::size_t v1, std::size_t v2) {
                return throw std::bad_alloc(), v1 + v2;
            },
            std::size_t(0));

        PIKA_TEST(false);
    }
    catch (std::bad_alloc const&)
    {
        caught_exception = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        pika::future<void> f = pika::inclusive_scan(
            p, iterator(std::begin(c)), iterator(std::end(c)), std::begin(d),
            [](std::size_t v1, std::size_t v2) {
                return throw std::bad_alloc(), v1 + v2;
            },
            std::size_t(0));

        returned_from_algorithm = true;
        f.get();

        PIKA_TEST(false);
    }
    catch (std::bad_alloc const&)
    {
        caught_exception = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);
    PIKA_TEST(returned_from_algorithm);
}

#define FILL_VALUE 10
#define ARRAY_SIZE 10000

// n'th value of sum of 1+2+3+...
int check_n_triangle(int n)
{
    return n < 0 ? 0 : (n) * (n + 1) / 2;
}

// n'th value of sum of x+x+x+...
int check_n_const(int n, int x)
{
    return n < 0 ? 0 : n * x;
}

// run scan algorithm, validate that output array hold expected answers.
template <typename ExPolicy>
void test_inclusive_scan_validate(
    ExPolicy p, std::vector<int>& a, std::vector<int>& b)
{
    using namespace pika::execution;

    // test 1, fill array with numbers counting from 0, then run scan algorithm
    a.clear();
    std::copy(pika::util::counting_iterator<int>(0),
        pika::util::counting_iterator<int>(ARRAY_SIZE), std::back_inserter(a));
    b.resize(a.size());
    pika::inclusive_scan(
        p, a.begin(), a.end(), b.begin(),
        [](int bar, int baz) { return bar + baz; }, 0);
    //
    for (int i = 0; i < static_cast<int>(b.size()); ++i)
    {
        // counting from zero,
        int value = b[i];    //-V108
        int expected_value = check_n_triangle(i);
        if (!PIKA_TEST_EQ(value, expected_value))
            break;
    }

    // test 2, fill array with numbers counting from 1, then run scan algorithm
    a.clear();
    std::copy(pika::util::counting_iterator<int>(1),
        pika::util::counting_iterator<int>(ARRAY_SIZE), std::back_inserter(a));
    b.resize(a.size());
    pika::inclusive_scan(
        p, a.begin(), a.end(), b.begin(),
        [](int bar, int baz) { return bar + baz; }, 0);
    //
    for (int i = 0; i < static_cast<int>(b.size()); ++i)
    {
        // counting from 1, use i+1
        int value = b[i];    //-V108
        int expected_value = check_n_triangle(i + 1);
        PIKA_TEST_EQ(value, expected_value);
        if (value != expected_value)
            break;
    }

    // test 3, fill array with constant
    a.clear();
    std::fill_n(std::back_inserter(a), ARRAY_SIZE, FILL_VALUE);
    b.resize(a.size());
    pika::inclusive_scan(
        p, a.begin(), a.end(), b.begin(),
        [](int bar, int baz) { return bar + baz; }, 0);
    //
    for (int i = 0; i < static_cast<int>(b.size()); ++i)
    {
        int value = b[i];    //-V108
        int expected_value = check_n_const(i + 1, FILL_VALUE);
        PIKA_TEST_EQ(value, expected_value);
        if (value != expected_value)
            break;
    }
}
