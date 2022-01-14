//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2015 Daniel Bourgeois
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/execution.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/container_algorithms/is_sorted.hpp>

#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////////
int seed = std::random_device{}();
std::mt19937 gen(seed);

using identity = pika::parallel::util::projection_identity;

struct negate
{
    int operator()(int x)
    {
        return -x;
    }
};

template <typename ExPolicy, typename IteratorTag>
void test_sorted1(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c), std::end(c), 0);

    PIKA_TEST(pika::ranges::is_sorted(std::forward<ExPolicy>(policy),
        iterator(std::begin(c)), iterator(std::end(c)), std::less<int>(),
        identity()));
    PIKA_TEST(!pika::ranges::is_sorted(std::forward<ExPolicy>(policy),
        iterator(std::begin(c)), iterator(std::end(c)), std::greater<int>(),
        identity()));
    PIKA_TEST(!pika::ranges::is_sorted(std::forward<ExPolicy>(policy),
        iterator(std::begin(c)), iterator(std::end(c)), std::less<int>(),
        negate()));
    PIKA_TEST(pika::ranges::is_sorted(std::forward<ExPolicy>(policy),
        iterator(std::begin(c)), iterator(std::end(c)), std::greater<int>(),
        negate()));
}

template <typename ExPolicy, typename IteratorTag>
void test_sorted1_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c), std::end(c), 0);

    pika::future<bool> f1 = pika::ranges::is_sorted(std::forward<ExPolicy>(p),
        iterator(std::begin(c)), iterator(std::end(c)), std::less<int>(),
        identity());
    f1.wait();
    PIKA_TEST(f1.get());
    pika::future<bool> f2 = pika::ranges::is_sorted(std::forward<ExPolicy>(p),
        iterator(std::begin(c)), iterator(std::end(c)), std::greater<int>(),
        identity());
    f2.wait();
    PIKA_TEST(!f2.get());
    pika::future<bool> f3 = pika::ranges::is_sorted(std::forward<ExPolicy>(p),
        iterator(std::begin(c)), iterator(std::end(c)), std::less<int>(),
        negate());
    f3.wait();
    PIKA_TEST(!f3.get());
    pika::future<bool> f4 = pika::ranges::is_sorted(std::forward<ExPolicy>(p),
        iterator(std::begin(c)), iterator(std::end(c)), std::greater<int>(),
        negate());
    f4.wait();
    PIKA_TEST(f4.get());
}

template <typename IteratorTag>
void test_sorted1_seq(IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c), std::end(c), 0);

    PIKA_TEST(pika::ranges::is_sorted(iterator(std::begin(c)),
        iterator(std::end(c)), std::less<int>(), identity()));
    PIKA_TEST(!pika::ranges::is_sorted(iterator(std::begin(c)),
        iterator(std::end(c)), std::greater<int>(), identity()));
    PIKA_TEST(!pika::ranges::is_sorted(iterator(std::begin(c)),
        iterator(std::end(c)), std::less<int>(), negate()));
    PIKA_TEST(pika::ranges::is_sorted(iterator(std::begin(c)),
        iterator(std::end(c)), std::greater<int>(), negate()));
}

template <typename ExPolicy>
void test_sorted1(ExPolicy&& policy)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    std::vector<int> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c), std::end(c), 0);

    PIKA_TEST(pika::ranges::is_sorted(
        std::forward<ExPolicy>(policy), c, std::less<int>(), identity()));
    PIKA_TEST(!pika::ranges::is_sorted(
        std::forward<ExPolicy>(policy), c, std::greater<int>(), identity()));
    PIKA_TEST(!pika::ranges::is_sorted(
        std::forward<ExPolicy>(policy), c, std::less<int>(), negate()));
    PIKA_TEST(pika::ranges::is_sorted(
        std::forward<ExPolicy>(policy), c, std::greater<int>(), negate()));
}

template <typename ExPolicy>
void test_sorted1_async(ExPolicy&& p)
{
    std::vector<int> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c), std::end(c), 0);

    pika::future<bool> f1 = pika::ranges::is_sorted(
        std::forward<ExPolicy>(p), c, std::less<int>(), identity());
    f1.wait();
    PIKA_TEST(f1.get());
    pika::future<bool> f2 = pika::ranges::is_sorted(
        std::forward<ExPolicy>(p), c, std::greater<int>(), identity());
    f2.wait();
    PIKA_TEST(!f2.get());
    pika::future<bool> f3 = pika::ranges::is_sorted(
        std::forward<ExPolicy>(p), c, std::less<int>(), negate());
    f3.wait();
    PIKA_TEST(!f3.get());
    pika::future<bool> f4 = pika::ranges::is_sorted(
        std::forward<ExPolicy>(p), c, std::greater<int>(), negate());
    f4.wait();
    PIKA_TEST(f4.get());
}

void test_sorted1_seq()
{
    std::vector<int> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c), std::end(c), 0);

    PIKA_TEST(pika::ranges::is_sorted(c, std::less<int>(), identity()));
    PIKA_TEST(!pika::ranges::is_sorted(c, std::greater<int>(), identity()));
    PIKA_TEST(!pika::ranges::is_sorted(c, std::less<int>(), negate()));
    PIKA_TEST(pika::ranges::is_sorted(c, std::greater<int>(), negate()));
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_sorted2(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c), std::end(c), 0);
    //Add a certain large value in middle of array to ignore
    int ignore = 20000;
    c[c.size() / 2] = ignore;
    //Provide custom predicate to ignore the value of ignore
    //pred should return true when it is given something deemed not sorted
    auto pred = [&ignore](int ahead, int behind) {
        return behind > ahead && behind != ignore;
    };

    PIKA_TEST(pika::ranges::is_sorted(policy, iterator(std::begin(c)),
        iterator(std::end(c)), pred, identity()));
    PIKA_TEST(!pika::ranges::is_sorted(policy, iterator(std::begin(c)),
        iterator(std::end(c)), pred,
        [ignore](int x) { return x == ignore ? -x : x; }));
}

template <typename ExPolicy, typename IteratorTag>
void test_sorted2_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c), std::end(c), 0);
    //Add a certain large value in middle of array to ignore
    int ignore = 20000;
    c[c.size() / 2] = ignore;
    //Provide custom predicate to ignore the value of ignore
    //pred should return true when it is given something deemed not sorted
    auto pred = [&ignore](int ahead, int behind) {
        return behind > ahead && behind != ignore;
    };

    pika::future<bool> f1 = pika::ranges::is_sorted(
        p, iterator(std::begin(c)), iterator(std::end(c)), pred);
    f1.wait();
    PIKA_TEST(f1.get());
    pika::future<bool> f2 = pika::ranges::is_sorted(p, iterator(std::begin(c)),
        iterator(std::end(c)), pred,
        [ignore](int x) { return x == ignore ? -x : x; });
    f2.wait();
    PIKA_TEST(!f2.get());
}

template <typename IteratorTag>
void test_sorted2_seq(IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c), std::end(c), 0);
    //Add a certain large value in middle of array to ignore
    int ignore = 20000;
    c[c.size() / 2] = ignore;
    //Provide custom predicate to ignore the value of ignore
    //pred should return true when it is given something deemed not sorted
    auto pred = [&ignore](int ahead, int behind) {
        return behind > ahead && behind != ignore;
    };

    PIKA_TEST(pika::ranges::is_sorted(
        iterator(std::begin(c)), iterator(std::end(c)), pred));
    PIKA_TEST(
        !pika::ranges::is_sorted(iterator(std::begin(c)), iterator(std::end(c)),
            pred, [ignore](int x) { return x == ignore ? -x : x; }));
}

template <typename ExPolicy>
void test_sorted2(ExPolicy policy)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    std::vector<int> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c), std::end(c), 0);
    //Add a certain large value in middle of array to ignore
    int ignore = 20000;
    c[c.size() / 2] = ignore;
    //Provide custom predicate to ignore the value of ignore
    //pred should return true when it is given something deemed not sorted
    auto pred = [&ignore](int ahead, int behind) {
        return behind > ahead && behind != ignore;
    };

    PIKA_TEST(pika::ranges::is_sorted(policy, c, pred));
    PIKA_TEST(!pika::ranges::is_sorted(
        policy, c, pred, [ignore](int x) { return x == ignore ? -x : x; }));
}

template <typename ExPolicy>
void test_sorted2_async(ExPolicy p)
{
    std::vector<int> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c), std::end(c), 0);
    //Add a certain large value in middle of array to ignore
    int ignore = 20000;
    c[c.size() / 2] = ignore;
    //Provide custom predicate to ignore the value of ignore
    //pred should return true when it is given something deemed not sorted
    auto pred = [&ignore](int ahead, int behind) {
        return behind > ahead && behind != ignore;
    };

    pika::future<bool> f1 = pika::ranges::is_sorted(p, c, pred);
    f1.wait();
    PIKA_TEST(f1.get());
    pika::future<bool> f2 = pika::ranges::is_sorted(
        p, c, pred, [ignore](int x) { return x == ignore ? -x : x; });
    f2.wait();
    PIKA_TEST(!f2.get());
}

void test_sorted2_seq()
{
    std::vector<int> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c), std::end(c), 0);
    //Add a certain large value in middle of array to ignore
    int ignore = 20000;
    c[c.size() / 2] = ignore;
    //Provide custom predicate to ignore the value of ignore
    //pred should return true when it is given something deemed not sorted
    auto pred = [&ignore](int ahead, int behind) {
        return behind > ahead && behind != ignore;
    };

    PIKA_TEST(pika::ranges::is_sorted(c, pred));
    PIKA_TEST(!pika::ranges::is_sorted(
        c, pred, [ignore](int x) { return x == ignore ? -x : x; }));
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_sorted3(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c_beg(10007);
    std::vector<int> c_end(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c_beg), std::end(c_beg), 0);
    std::iota(std::begin(c_end), std::end(c_end), 0);
    //add unsorted element to c_beg, c_end at the beginning, end respectively
    c_beg[0] = 20000;
    c_end[c_end.size() - 1] = -20000;

    bool is_ordered1 = pika::ranges::is_sorted(
        policy, iterator(std::begin(c_beg)), iterator(std::end(c_beg)));
    bool is_ordered2 = pika::ranges::is_sorted(
        policy, iterator(std::begin(c_end)), iterator(std::end(c_end)));
    bool is_ordered3 = pika::ranges::is_sorted(policy,
        iterator(std::begin(c_beg)), iterator(std::end(c_beg)),
        std::less<int>(), [](int x) { return x == 20000 ? 0 : x; });
    bool is_ordered4 = pika::ranges::is_sorted(policy,
        iterator(std::begin(c_end)), iterator(std::end(c_end)),
        std::less<int>(), [](int x) { return x == -20000 ? 10006 : x; });

    PIKA_TEST(!is_ordered1);
    PIKA_TEST(!is_ordered2);
    PIKA_TEST(is_ordered3);
    PIKA_TEST(is_ordered4);
}

template <typename ExPolicy, typename IteratorTag>
void test_sorted3_async(ExPolicy p, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c_beg(10007);
    std::vector<int> c_end(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c_beg), std::end(c_beg), 0);
    std::iota(std::begin(c_end), std::end(c_end), 0);
    //add unsorted element to c_beg, c_end at the beginning, end respectively
    c_beg[0] = 20000;
    c_end[c_end.size() - 1] = -20000;

    pika::future<bool> f1 = pika::ranges::is_sorted(
        p, iterator(std::begin(c_beg)), iterator(std::end(c_beg)));
    pika::future<bool> f2 = pika::ranges::is_sorted(
        p, iterator(std::begin(c_end)), iterator(std::end(c_end)));
    pika::future<bool> f3 = pika::ranges::is_sorted(p,
        iterator(std::begin(c_beg)), iterator(std::end(c_beg)),
        std::less<int>(), [](int x) { return x == 20000 ? 0 : x; });
    pika::future<bool> f4 = pika::ranges::is_sorted(p,
        iterator(std::begin(c_end)), iterator(std::end(c_end)),
        std::less<int>(), [](int x) { return x == -20000 ? 10006 : x; });
    f1.wait();
    PIKA_TEST(!f1.get());
    f2.wait();
    PIKA_TEST(!f2.get());
    f3.wait();
    PIKA_TEST(f3.get());
    f4.wait();
    PIKA_TEST(f4.get());
}

template <typename IteratorTag>
void test_sorted3_seq(IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c_beg(10007);
    std::vector<int> c_end(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c_beg), std::end(c_beg), 0);
    std::iota(std::begin(c_end), std::end(c_end), 0);
    //add unsorted element to c_beg, c_end at the beginning, end respectively
    c_beg[0] = 20000;
    c_end[c_end.size() - 1] = -20000;

    PIKA_TEST(!pika::ranges::is_sorted(
        iterator(std::begin(c_beg)), iterator(std::end(c_beg))));
    PIKA_TEST(!pika::ranges::is_sorted(
        iterator(std::begin(c_end)), iterator(std::end(c_end))));
    PIKA_TEST(pika::ranges::is_sorted(iterator(std::begin(c_beg)),
        iterator(std::end(c_beg)), std::less<int>(),
        [](int x) { return x == 20000 ? 0 : x; }));
    PIKA_TEST(pika::ranges::is_sorted(iterator(std::begin(c_end)),
        iterator(std::end(c_end)), std::less<int>(),
        [](int x) { return x == -20000 ? 10006 : x; }));
}

template <typename ExPolicy>
void test_sorted3(ExPolicy policy)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    std::vector<int> c_beg(10007);
    std::vector<int> c_end(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c_beg), std::end(c_beg), 0);
    std::iota(std::begin(c_end), std::end(c_end), 0);
    //add unsorted element to c_beg, c_end at the beginning, end respectively
    c_beg[0] = 20000;
    c_end[c_end.size() - 1] = -20000;

    PIKA_TEST(!pika::ranges::is_sorted(policy, c_beg));
    PIKA_TEST(!pika::ranges::is_sorted(policy, c_end));
    PIKA_TEST(pika::ranges::is_sorted(policy, c_beg, std::less<int>(),
        [](int x) { return x == 20000 ? 0 : x; }));
    PIKA_TEST(pika::ranges::is_sorted(policy, c_end, std::less<int>(),
        [](int x) { return x == -20000 ? 10006 : x; }));
}

template <typename ExPolicy>
void test_sorted3_async(ExPolicy p)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    std::vector<int> c_beg(10007);
    std::vector<int> c_end(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c_beg), std::end(c_beg), 0);
    std::iota(std::begin(c_end), std::end(c_end), 0);
    //add unsorted element to c_beg, c_end at the beginning, end respectively
    c_beg[0] = 20000;
    c_end[c_end.size() - 1] = -20000;

    pika::future<bool> f1 = pika::ranges::is_sorted(p, c_beg);
    pika::future<bool> f2 = pika::ranges::is_sorted(p, c_end);
    pika::future<bool> f3 = pika::ranges::is_sorted(
        p, c_beg, std::less<int>(), [](int x) { return x == 20000 ? 0 : x; });
    pika::future<bool> f4 = pika::ranges::is_sorted(p, c_end, std::less<int>(),
        [](int x) { return x == -20000 ? 10006 : x; });

    f1.wait();
    PIKA_TEST(!f1.get());
    f2.wait();
    PIKA_TEST(!f2.get());
    f3.wait();
    PIKA_TEST(f3.get());
    f4.wait();
    PIKA_TEST(f4.get());
}

void test_sorted3_seq()
{
    std::vector<int> c_beg(10007);
    std::vector<int> c_end(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c_beg), std::end(c_beg), 0);
    std::iota(std::begin(c_end), std::end(c_end), 0);
    //add unsorted element to c_beg, c_end at the beginning, end respectively
    c_beg[0] = 20000;
    c_end[c_end.size() - 1] = -20000;

    PIKA_TEST(!pika::ranges::is_sorted(c_beg));
    PIKA_TEST(!pika::ranges::is_sorted(c_end));
    PIKA_TEST(pika::ranges::is_sorted(
        c_beg, std::less<int>(), [](int x) { return x == 20000 ? 0 : x; }));
    PIKA_TEST(pika::ranges::is_sorted(c_end, std::less<int>(),
        [](int x) { return x == -20000 ? 10006 : x; }));
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_sorted_exception(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), 0);

    bool caught_exception = false;
    try
    {
        pika::ranges::is_sorted(policy,
            decorated_iterator(
                std::begin(c), []() { throw std::runtime_error("test"); }),
            decorated_iterator(
                std::end(c), []() { throw std::runtime_error("test"); }));
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

    caught_exception = false;
    try
    {
        pika::ranges::is_sorted(policy, iterator(std::begin(c)),
            iterator(std::end(c)),
            [](int, int) -> bool { throw std::runtime_error("test"); });
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

    caught_exception = false;
    try
    {
        pika::ranges::is_sorted(policy, iterator(std::begin(c)),
            iterator(std::end(c)), std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
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
void test_sorted_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen() + 1);

    bool caught_exception = false;
    try
    {
        pika::future<bool> f = pika::ranges::is_sorted(p,
            decorated_iterator(
                std::begin(c), []() { throw std::runtime_error("test"); }),
            decorated_iterator(
                std::end(c), []() { throw std::runtime_error("test"); }));
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

    caught_exception = false;
    try
    {
        pika::future<bool> f = pika::ranges::is_sorted(p, iterator(std::begin(c)),
            iterator(std::end(c)),
            [](int, int) -> bool { throw std::runtime_error("test"); });
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

    PIKA_TEST(caught_exception);

    caught_exception = false;
    try
    {
        pika::future<bool> f = pika::ranges::is_sorted(p, iterator(std::begin(c)),
            iterator(std::end(c)), std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
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
}

template <typename IteratorTag>
void test_sorted_exception_seq(IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), 0);

    bool caught_exception = false;
    try
    {
        pika::ranges::is_sorted(decorated_iterator(std::begin(c),
                                   []() { throw std::runtime_error("test"); }),
            decorated_iterator(
                std::end(c), []() { throw std::runtime_error("test"); }));
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

    caught_exception = false;
    try
    {
        pika::ranges::is_sorted(iterator(std::begin(c)), iterator(std::end(c)),
            [](int, int) -> bool { throw std::runtime_error("test"); });
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

    caught_exception = false;
    try
    {
        pika::ranges::is_sorted(iterator(std::begin(c)), iterator(std::end(c)),
            std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
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

template <typename ExPolicy>
void test_sorted_exception(ExPolicy policy)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), 0);

    bool caught_exception = false;
    try
    {
        pika::ranges::is_sorted(policy, c,
            [](int, int) -> bool { throw std::runtime_error("test"); });
    }
    catch (pika::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions_base<ExPolicy>::call(policy, e);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);

    caught_exception = false;
    try
    {
        pika::ranges::is_sorted(policy, c, std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
    }
    catch (pika::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions_base<ExPolicy>::call(policy, e);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);
}

template <typename ExPolicy>
void test_sorted_exception_async(ExPolicy p)
{
    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen() + 1);

    bool caught_exception = false;
    try
    {
        pika::future<bool> f = pika::ranges::is_sorted(
            p, c, [](int, int) -> bool { throw std::runtime_error("test"); });
        f.get();

        PIKA_TEST(false);
    }
    catch (pika::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions_base<ExPolicy>::call(p, e);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);

    caught_exception = false;
    try
    {
        pika::future<bool> f = pika::ranges::is_sorted(p, c, std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
        f.get();

        PIKA_TEST(false);
    }
    catch (pika::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions_base<ExPolicy>::call(p, e);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);
}

void test_sorted_exception_seq()
{
    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), 0);

    bool caught_exception = false;
    try
    {
        pika::ranges::is_sorted(
            c, [](int, int) -> int { throw std::runtime_error("test"); });
    }
    catch (pika::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions_base<pika::execution::sequenced_policy>::call(
            pika::execution::seq, e);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);

    caught_exception = false;
    try
    {
        pika::ranges::is_sorted(c, std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
    }
    catch (pika::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions_base<pika::execution::sequenced_policy>::call(
            pika::execution::seq, e);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_sorted_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), 0);

    bool caught_bad_alloc = false;
    try
    {
        pika::ranges::is_sorted(policy,
            decorated_iterator(
                std::begin(c), []() { throw std::runtime_error("test"); }),
            decorated_iterator(
                std::end(c), []() { throw std::runtime_error("test"); }));
    }
    catch (pika::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_bad_alloc);

    caught_bad_alloc = false;
    try
    {
        pika::ranges::is_sorted(policy, iterator(std::begin(c)),
            iterator(std::end(c)),
            [](int, int) -> bool { throw std::runtime_error("test"); });
    }
    catch (pika::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_bad_alloc);

    caught_bad_alloc = false;
    try
    {
        pika::ranges::is_sorted(policy, iterator(std::begin(c)),
            iterator(std::end(c)), std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
    }
    catch (pika::exception_list const& e)
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
void test_sorted_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen() + 1);

    bool caught_bad_alloc = false;
    try
    {
        pika::future<bool> f = pika::ranges::is_sorted(p,
            decorated_iterator(
                std::begin(c), []() { throw std::runtime_error("test"); }),
            decorated_iterator(
                std::end(c), []() { throw std::runtime_error("test"); }));
        f.get();

        PIKA_TEST(false);
    }
    catch (pika::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_bad_alloc);

    caught_bad_alloc = false;
    try
    {
        pika::future<bool> f = pika::ranges::is_sorted(p, iterator(std::begin(c)),
            iterator(std::end(c)),
            [](int, int) -> bool { throw std::runtime_error("test"); });
        f.get();

        PIKA_TEST(false);
    }
    catch (pika::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_bad_alloc);

    PIKA_TEST(caught_bad_alloc);

    caught_bad_alloc = false;
    try
    {
        pika::future<bool> f = pika::ranges::is_sorted(p, iterator(std::begin(c)),
            iterator(std::end(c)), std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
        f.get();

        PIKA_TEST(false);
    }
    catch (pika::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_bad_alloc);
}

template <typename IteratorTag>
void test_sorted_bad_alloc_seq(IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), 0);

    bool caught_bad_alloc = false;
    try
    {
        pika::ranges::is_sorted(decorated_iterator(std::begin(c),
                                   []() { throw std::runtime_error("test"); }),
            decorated_iterator(
                std::end(c), []() { throw std::runtime_error("test"); }));
    }
    catch (pika::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_bad_alloc);

    caught_bad_alloc = false;
    try
    {
        pika::ranges::is_sorted(iterator(std::begin(c)), iterator(std::end(c)),
            [](int, int) -> bool { throw std::runtime_error("test"); });
    }
    catch (pika::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_bad_alloc);

    caught_bad_alloc = false;
    try
    {
        pika::ranges::is_sorted(iterator(std::begin(c)), iterator(std::end(c)),
            std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
    }
    catch (pika::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_bad_alloc);
}

template <typename ExPolicy>
void test_sorted_bad_alloc(ExPolicy policy)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), 0);

    bool caught_bad_alloc = false;
    try
    {
        pika::ranges::is_sorted(policy, c,
            [](int, int) -> bool { throw std::runtime_error("test"); });
    }
    catch (pika::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_bad_alloc);

    caught_bad_alloc = false;
    try
    {
        pika::ranges::is_sorted(policy, c, std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
    }
    catch (pika::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_bad_alloc);
}

template <typename ExPolicy>
void test_sorted_bad_alloc_async(ExPolicy p)
{
    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen() + 1);

    bool caught_bad_alloc = false;
    try
    {
        pika::future<bool> f = pika::ranges::is_sorted(
            p, c, [](int, int) -> bool { throw std::runtime_error("test"); });
        f.get();

        PIKA_TEST(false);
    }
    catch (pika::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_bad_alloc);

    caught_bad_alloc = false;
    try
    {
        pika::future<bool> f = pika::ranges::is_sorted(p, c, std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
        f.get();

        PIKA_TEST(false);
    }
    catch (pika::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_bad_alloc);
}

void test_sorted_bad_alloc_seq()
{
    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), 0);

    bool caught_bad_alloc = false;
    try
    {
        pika::ranges::is_sorted(
            c, [](int, int) -> int { throw std::runtime_error("test"); });
    }
    catch (pika::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_bad_alloc);

    caught_bad_alloc = false;
    try
    {
        pika::ranges::is_sorted(c, std::less<int>(),
            [](int) -> int { throw std::runtime_error("test"); });
    }
    catch (pika::exception_list const& e)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_bad_alloc);
}
