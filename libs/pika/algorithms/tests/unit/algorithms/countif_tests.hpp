//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/modules/testing.hpp>
#include <pika/parallel/algorithms/count.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "test_utils.hpp"

//////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);
std::uniform_int_distribution<> dis(0, (std::numeric_limits<int>::max)());

struct smaller_than_50
{
    template <typename T>
    auto operator()(T const& x) const -> decltype(x < 50)
    {
        return x < 50;
    }
};

struct always_true
{
    template <typename T>
    bool operator()(T const&) const
    {
        return true;
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_count_if(IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef std::vector<int>::difference_type diff_type;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(100007);
    std::iota(std::begin(c), std::begin(c) + 50, 0);
    std::iota(std::begin(c) + 50, std::end(c), dis(gen) + 50);

    diff_type num_items = pika::count_if(
        iterator(std::begin(c)), iterator(std::end(c)), smaller_than_50());

    PIKA_TEST_EQ(num_items, 50u);
}

template <typename ExPolicy, typename IteratorTag>
void test_count_if(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef std::vector<int>::difference_type diff_type;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(100007);
    std::iota(std::begin(c), std::begin(c) + 50, 0);
    std::iota(std::begin(c) + 50, std::end(c), dis(gen) + 50);

    diff_type num_items = pika::count_if(policy, iterator(std::begin(c)),
        iterator(std::end(c)), smaller_than_50());

    PIKA_TEST_EQ(num_items, 50u);
}

template <typename ExPolicy, typename IteratorTag>
void test_count_if_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef std::vector<int>::difference_type diff_type;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::begin(c) + 50, 0);
    std::iota(std::begin(c) + 50, std::end(c), dis(gen) + 50);

    pika::future<diff_type> f = pika::count_if(
        p, iterator(std::begin(c)), iterator(std::end(c)), smaller_than_50());

    PIKA_TEST_EQ(f.get(), 50);
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_count_if_exception(IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), dis(gen));

    bool caught_exception = false;
    try
    {
        // pred should never proc, so simple 'returns true'
        pika::count_if(decorated_iterator(std::begin(c),
                          []() { throw std::runtime_error("test"); }),
            decorated_iterator(std::end(c)), always_true());
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
void test_count_if_exception(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), dis(gen));

    bool caught_exception = false;
    try
    {
        // pred should never proc, so simple 'returns true'
        pika::count_if(policy,
            decorated_iterator(
                std::begin(c), []() { throw std::runtime_error("test"); }),
            decorated_iterator(std::end(c)), always_true());
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
void test_count_if_exception_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef std::vector<int>::difference_type diff_type;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<int> c(10007);
    std::fill(std::begin(c), std::end(c), 10);

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        pika::future<diff_type> f = pika::count_if(p,
            decorated_iterator(
                std::begin(c), []() { throw std::runtime_error("test"); }),
            decorated_iterator(std::end(c)), always_true());
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
void test_count_if_bad_alloc(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), dis(gen));

    bool caught_bad_alloc = false;
    try
    {
        pika::count_if(policy,
            decorated_iterator(std::begin(c), []() { throw std::bad_alloc(); }),
            decorated_iterator(std::end(c)), always_true());
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
void test_count_if_bad_alloc_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef std::vector<int>::difference_type diff_type;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), dis(gen));

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try
    {
        pika::future<diff_type> f = pika::count_if(p,
            decorated_iterator(std::begin(c), []() { throw std::bad_alloc(); }),
            decorated_iterator(std::end(c)), always_true());
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
