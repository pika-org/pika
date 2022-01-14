//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/iterator_support/iterator_range.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/container_algorithms/for_each.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////////
struct counter
{
    std::size_t count = 0;
    void operator()(std::size_t& v)
    {
        ++count;
        v = 42;
    }
};

template <typename IteratorTag>
void test_for_each_seq(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    counter f;
    auto res = pika::ranges::for_each(
        pika::util::make_iterator_range(
            iterator(std::begin(c)), iterator(std::end(c))),
        f);

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](std::size_t v) -> void {
        PIKA_TEST_EQ(v, std::size_t(42));
        ++count;
    });
    PIKA_TEST_EQ(count, c.size());
    PIKA_TEST(res.in == iterator(std::end(c)));
    PIKA_TEST_EQ(res.fun.count, c.size());
    PIKA_TEST_EQ(f.count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_for_each(ExPolicy&& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(pika::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    iterator result = pika::ranges::for_each(std::forward<ExPolicy>(policy),
        pika::util::make_iterator_range(
            iterator(std::begin(c)), iterator(std::end(c))),
        [](std::size_t& v) { v = 42; });
    PIKA_TEST(result == iterator(std::end(c)));

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](std::size_t v) -> void {
        PIKA_TEST_EQ(v, std::size_t(42));
        ++count;
    });
    PIKA_TEST_EQ(count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_for_each_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    pika::future<iterator> f = pika::ranges::for_each(std::forward<ExPolicy>(p),
        pika::util::make_iterator_range(
            iterator(std::begin(c)), iterator(std::end(c))),
        [](std::size_t& v) { v = 42; });
    PIKA_TEST(f.get() == iterator(std::end(c)));

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](std::size_t v) -> void {
        PIKA_TEST_EQ(v, std::size_t(42));
        ++count;
    });
    PIKA_TEST_EQ(count, c.size());
}

////////////////////////////////////////////////////////////////////////////////
struct counter_exception
{
    std::size_t count = 0;
    void operator()(std::size_t&)
    {
        ++count;
        throw std::runtime_error("test");
    }
};

template <typename IteratorTag>
void test_for_each_exception_seq(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    counter_exception f;
    try
    {
        pika::ranges::for_each(
            pika::util::make_iterator_range(
                iterator(std::begin(c)), iterator(std::end(c))),
            f);

        PIKA_TEST(false);
    }
    catch (pika::exception_list const& e)
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
    BOOST_STATIC_ASSERT(pika::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    try
    {
        pika::ranges::for_each(policy,
            pika::util::make_iterator_range(
                iterator(std::begin(c)), iterator(std::end(c))),
            [](std::size_t&) { throw std::runtime_error("test"); });

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
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        pika::future<iterator> f = pika::ranges::for_each(p,
            pika::util::make_iterator_range(
                iterator(std::begin(c)), iterator(std::end(c))),
            [](std::size_t&) { throw std::runtime_error("test"); });
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

////////////////////////////////////////////////////////////////////////////////
struct counter_bad_alloc
{
    std::size_t count = 0;
    void operator()(std::size_t&)
    {
        ++count;
        throw std::bad_alloc();
    }
};

template <typename IteratorTag>
void test_for_each_bad_alloc_seq(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    counter_bad_alloc f;
    try
    {
        pika::ranges::for_each(
            pika::util::make_iterator_range(
                iterator(std::begin(c)), iterator(std::end(c))),
            f);

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
    BOOST_STATIC_ASSERT(pika::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    try
    {
        pika::ranges::for_each(policy,
            pika::util::make_iterator_range(
                iterator(std::begin(c)), iterator(std::end(c))),
            [](std::size_t&) { throw std::bad_alloc(); });

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
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        pika::future<iterator> f = pika::ranges::for_each(p,
            pika::util::make_iterator_range(
                iterator(std::begin(c)), iterator(std::end(c))),
            [](std::size_t&) { throw std::bad_alloc(); });
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

template <typename ExPolicy, typename IteratorTag>
void test_for_each_sender(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    namespace ex = pika::execution::experimental;

    auto rng = pika::util::make_iterator_range(
        iterator(std::begin(c)), iterator(std::end(c)));
    auto f = [](std::size_t& v) { v = 42; };
    auto result = ex::just(rng, f) |
        pika::ranges::for_each(std::forward<ExPolicy>(p)) | ex::sync_wait();
    PIKA_TEST(result == iterator(std::end(c)));

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](std::size_t v) -> void {
        PIKA_TEST_EQ(v, std::size_t(42));
        ++count;
    });
    PIKA_TEST_EQ(count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_for_each_exception_sender(ExPolicy p, IteratorTag)
{
    namespace ex = pika::execution::experimental;

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    auto rng = pika::util::make_iterator_range(
        iterator(std::begin(c)), iterator(std::end(c)));
    auto f = [](std::size_t&) { throw std::runtime_error("test"); };

    bool caught_exception = false;
    try
    {
        ex::just(rng, f) | pika::ranges::for_each(std::forward<ExPolicy>(p)) |
            ex::sync_wait();

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
}

template <typename ExPolicy, typename IteratorTag>
void test_for_each_bad_alloc_sender(ExPolicy p, IteratorTag)
{
    namespace ex = pika::execution::experimental;

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    auto rng = pika::util::make_iterator_range(
        iterator(std::begin(c)), iterator(std::end(c)));
    auto f = [](std::size_t&) { throw std::bad_alloc(); };

    bool caught_exception = false;
    try
    {
        ex::just(rng, f) | pika::ranges::for_each(std::forward<ExPolicy>(p)) |
            ex::sync_wait();

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
