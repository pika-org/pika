//  Copyright (c) 2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/execution.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/algorithms/for_each.hpp>

#include <cstddef>
#include <iterator>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
int seed = std::random_device{}();
std::mt19937 gen(seed);

struct set_42
{
    std::size_t count = 0;
    template <typename T>
    void operator()(T& val)
    {
        ++count;
        val = T(42);
    }
};

struct throw_always
{
    std::size_t count = 0;
    template <typename T>
    void operator()(T)
    {
        ++count;
        throw std::runtime_error("test");
    }
};

struct throw_bad_alloc
{
    std::size_t count = 0;
    template <typename T>
    void operator()(T)
    {
        ++count;
        throw std::bad_alloc();
    }
};

////////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_for_each_seq(IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    set_42 f;
    auto res = pika::for_each(iterator(std::begin(c)), iterator(std::end(c)), f);

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](int v) -> void {
        PIKA_TEST_EQ(v, int(42));
        ++count;
    });
    PIKA_TEST_EQ(count, c.size());
    PIKA_TEST_EQ(res.count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_for_each(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    pika::for_each(std::forward<ExPolicy>(policy), iterator(std::begin(c)),
        iterator(std::end(c)), set_42());

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](int v) -> void {
        PIKA_TEST_EQ(v, int(42));
        ++count;
    });
    PIKA_TEST_EQ(count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_for_each_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    pika::future<void> f = pika::for_each(std::forward<ExPolicy>(p),
        iterator(std::begin(c)), iterator(std::end(c)), set_42());
    f.get();

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](int v) -> void {
        PIKA_TEST_EQ(v, int(42));
        ++count;
    });
    PIKA_TEST_EQ(count, c.size());
}

////////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_for_each_exception_seq(IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    bool caught_exception = false;
    throw_always f;
    try
    {
        pika::for_each(iterator(std::begin(c)), iterator(std::end(c)), f);

        PIKA_TEST(false);
    }
    catch (pika::exception_list const&)
    {
        caught_exception = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);
    PIKA_TEST_EQ(f.count, std::size_t(1));
}

template <typename ExPolicy, typename IteratorTag>
void test_for_each_exception(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    bool caught_exception = false;
    try
    {
        pika::for_each(policy, iterator(std::begin(c)), iterator(std::end(c)),
            throw_always());

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
void test_for_each_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        pika::future<void> f = pika::for_each(
            p, iterator(std::begin(c)), iterator(std::end(c)), throw_always());
        returned_from_algorithm = true;
        f.get();

        PIKA_TEST(false);
    }
    catch (pika::exception_list const&)
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

////////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_for_each_bad_alloc_seq(IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    bool caught_exception = false;
    throw_bad_alloc f;
    try
    {
        pika::for_each(iterator(std::begin(c)), iterator(std::end(c)), f);

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
    PIKA_TEST_EQ(f.count, std::size_t(1));
}

template <typename ExPolicy, typename IteratorTag>
void test_for_each_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    bool caught_exception = false;
    try
    {
        pika::for_each(policy, iterator(std::begin(c)), iterator(std::end(c)),
            throw_bad_alloc());

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
void test_for_each_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        pika::future<void> f = pika::for_each(p, iterator(std::begin(c)),
            iterator(std::end(c)), throw_bad_alloc());
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

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_for_each_n_seq(IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    iterator result =
        pika::for_each_n(iterator(std::begin(c)), c.size(), set_42());
    iterator end = iterator(std::end(c));
    PIKA_TEST(result == end);

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](int v) -> void {
        PIKA_TEST_EQ(v, int(42));
        ++count;
    });
    PIKA_TEST_EQ(count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_for_each_n(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    iterator result =
        pika::for_each_n(policy, iterator(std::begin(c)), c.size(), set_42());
    iterator end = iterator(std::end(c));
    PIKA_TEST(result == end);

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](int v) -> void {
        PIKA_TEST_EQ(v, int(42));
        ++count;
    });
    PIKA_TEST_EQ(count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_for_each_n_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    pika::future<iterator> f =
        pika::for_each_n(p, iterator(std::begin(c)), c.size(), set_42());
    PIKA_TEST(f.get() == iterator(std::end(c)));

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](int v) -> void {
        PIKA_TEST_EQ(v, int(42));
        ++count;
    });
    PIKA_TEST_EQ(count, c.size());
}
