//  Copyright (c) 2014-2016 Hartmut Kaiser
//  Copyright (c) 2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/modules/testing.hpp>
#include <pika/parallel/algorithms/transform.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

struct add
{
    template <typename T1, typename T2>
    auto operator()(T1 const& v1, T2 const& v2) const -> decltype(v1 + v2)
    {
        return v1 + v2;
    }
};

struct throw_always
{
    template <typename T1, typename T2>
    auto operator()(T1 const& v1, T2 const& v2) const -> decltype(v1 + v2)
    {
        throw std::runtime_error("test");
    }
};

struct throw_bad_alloc
{
    template <typename T1, typename T2>
    auto operator()(T1 const& v1, T2 const& v2) const -> decltype(v1 + v2)
    {
        throw std::bad_alloc();
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_transform_binary(IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c1(10007);
    std::vector<int> c2(c1.size());
    std::vector<int> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1),
        std::rand() % ((std::numeric_limits<int>::max)() / 2));
    std::iota(std::begin(c2), std::end(c2),
        std::rand() % ((std::numeric_limits<int>::max)() / 2));

    auto result = pika::transform(iterator(std::begin(c1)),
        iterator(std::end(c1)), std::begin(c2), std::begin(d1), add());

    PIKA_TEST(result == std::end(d1));

    // verify values
    std::vector<int> d2(c1.size());
    std::transform(
        std::begin(c1), std::end(c1), std::begin(c2), std::begin(d2), add());

    std::size_t count = 0;
    PIKA_TEST(std::equal(std::begin(d1), std::end(d1), std::begin(d2),
        [&count](int v1, int v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    PIKA_TEST_EQ(count, d2.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_transform_binary(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c1(10007);
    std::vector<int> c2(c1.size());
    std::vector<int> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1),
        std::rand() % ((std::numeric_limits<int>::max)() / 2));
    std::iota(std::begin(c2), std::end(c2),
        std::rand() % ((std::numeric_limits<int>::max)() / 2));

    auto result = pika::transform(policy, iterator(std::begin(c1)),
        iterator(std::end(c1)), std::begin(c2), std::begin(d1), add());

    PIKA_TEST(result == std::end(d1));

    // verify values
    std::vector<int> d2(c1.size());
    std::transform(
        std::begin(c1), std::end(c1), std::begin(c2), std::begin(d2), add());

    std::size_t count = 0;
    PIKA_TEST(std::equal(std::begin(d1), std::end(d1), std::begin(d2),
        [&count](int v1, int v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    PIKA_TEST_EQ(count, d2.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_transform_binary_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c1(10007);
    std::vector<int> c2(c1.size());
    std::vector<int> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1),
        std::rand() % ((std::numeric_limits<int>::max)() / 2));
    std::iota(std::begin(c2), std::end(c2),
        std::rand() % ((std::numeric_limits<int>::max)() / 2));

    auto f = pika::transform(p, iterator(std::begin(c1)), iterator(std::end(c1)),
        std::begin(c2), std::begin(d1), add());
    f.wait();

    auto result = f.get();
    PIKA_TEST(result == std::end(d1));

    // verify values
    std::vector<int> d2(c1.size());
    std::transform(
        std::begin(c1), std::end(c1), std::begin(c2), std::begin(d2), add());

    std::size_t count = 0;
    PIKA_TEST(std::equal(std::begin(d1), std::end(d1), std::begin(d2),
        [&count](int v1, int v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    PIKA_TEST_EQ(count, d2.size());
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_transform_binary_exception(IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c1(10007);
    std::vector<int> c2(c1.size());
    std::vector<int> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1), std::rand());
    std::iota(std::begin(c2), std::end(c2), std::rand());

    bool caught_exception = false;
    try
    {
        pika::transform(iterator(std::begin(c1)), iterator(std::end(c1)),
            std::begin(c2), std::begin(d1), throw_always());

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
void test_transform_binary_exception(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c1(10007);
    std::vector<int> c2(c1.size());
    std::vector<int> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1), std::rand());
    std::iota(std::begin(c2), std::end(c2), std::rand());

    bool caught_exception = false;
    try
    {
        pika::transform(policy, iterator(std::begin(c1)), iterator(std::end(c1)),
            std::begin(c2), std::begin(d1), throw_always());

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
void test_transform_binary_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c1(10007);
    std::vector<int> c2(c1.size());
    std::vector<int> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1), std::rand());
    std::iota(std::begin(c2), std::end(c2), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f =
            pika::transform(p, iterator(std::begin(c1)), iterator(std::end(c1)),
                std::begin(c2), std::begin(d1), throw_always());
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
void test_transform_binary_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c1(10007);
    std::vector<int> c2(c1.size());
    std::vector<int> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1), std::rand());
    std::iota(std::begin(c2), std::end(c2), std::rand());

    bool caught_bad_alloc = false;
    try
    {
        pika::transform(policy, iterator(std::begin(c1)), iterator(std::end(c1)),
            std::begin(c2), std::begin(d1), throw_bad_alloc());

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
void test_transform_binary_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c1(10007);
    std::vector<int> c2(c1.size());
    std::vector<int> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1), std::rand());
    std::iota(std::begin(c2), std::end(c2), std::rand());

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f =
            pika::transform(p, iterator(std::begin(c1)), iterator(std::end(c1)),
                std::begin(c2), std::begin(d1), throw_bad_alloc());
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
