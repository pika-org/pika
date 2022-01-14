//  Copyright (C) 2012 Hartmut Kaiser
//  (C) Copyright 2008-10 Anthony Williams
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>

#include <chrono>
#include <deque>
#include <list>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int make_int_slowly()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return 42;
}

void test_wait_for_either_of_two_futures_1()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::future<int> f2(pt2.get_future());

    pt1();

    pika::future<
        pika::when_any_result<pika::tuple<pika::future<int>, pika::future<int>>>>
        r = pika::when_any(f1, f2);
    pika::tuple<pika::future<int>, pika::future<int>> t = r.get().futures;

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());

    PIKA_TEST(pika::get<0>(t).is_ready());
    PIKA_TEST_EQ(pika::get<0>(t).get(), 42);
}

void test_wait_for_either_of_two_futures_2()
{
    pika::lcos::local::packaged_task<int()> pt(make_int_slowly);
    pika::future<int> f1(pt.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::future<int> f2(pt2.get_future());

    pt2();

    pika::future<
        pika::when_any_result<pika::tuple<pika::future<int>, pika::future<int>>>>
        r = pika::when_any(f1, f2);
    pika::tuple<pika::future<int>, pika::future<int>> t = r.get().futures;

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());

    PIKA_TEST(pika::get<1>(t).is_ready());
    PIKA_TEST_EQ(pika::get<1>(t).get(), 42);
}

template <class Container>
void test_wait_for_either_of_two_futures_list_1()
{
    Container futures;
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    futures.push_back(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    futures.push_back(pt2.get_future());

    pt1();

    pika::future<pika::when_any_result<Container>> r = pika::when_any(futures);
    pika::when_any_result<Container> raw = r.get();

    PIKA_TEST_EQ(raw.index, 0u);

    Container t = std::move(raw.futures);

    PIKA_TEST(!futures.front().valid());
    PIKA_TEST(!futures.back().valid());

    PIKA_TEST(t.front().is_ready());
    PIKA_TEST_EQ(t.front().get(), 42);
}

template <class Container>
void test_wait_for_either_of_two_futures_list_2()
{
    Container futures;
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    futures.push_back(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    futures.push_back(pt2.get_future());

    pt2();

    pika::future<pika::when_any_result<Container>> r = pika::when_any(futures);
    pika::when_any_result<Container> raw = r.get();

    PIKA_TEST_EQ(raw.index, 1u);

    Container t = std::move(raw.futures);

    PIKA_TEST(!futures.front().valid());
    PIKA_TEST(!futures.back().valid());

    PIKA_TEST(t.back().is_ready());
    PIKA_TEST_EQ(t.back().get(), 42);
}

void test_wait_for_either_of_three_futures_1()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::future<int> f3(pt3.get_future());

    pt1();

    pika::future<pika::when_any_result<
        pika::tuple<pika::future<int>, pika::future<int>, pika::future<int>>>>
        r = pika::when_any(f1, f2, f3);
    pika::tuple<pika::future<int>, pika::future<int>, pika::future<int>> t =
        r.get().futures;

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());
    PIKA_TEST(!f3.valid());

    PIKA_TEST(pika::get<0>(t).is_ready());
    PIKA_TEST_EQ(pika::get<0>(t).get(), 42);
}

void test_wait_for_either_of_three_futures_2()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::future<int> f3(pt3.get_future());

    pt2();

    pika::future<pika::when_any_result<
        pika::tuple<pika::future<int>, pika::future<int>, pika::future<int>>>>
        r = pika::when_any(f1, f2, f3);
    pika::tuple<pika::future<int>, pika::future<int>, pika::future<int>> t =
        r.get().futures;

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());
    PIKA_TEST(!f3.valid());

    PIKA_TEST(pika::get<1>(t).is_ready());
    PIKA_TEST_EQ(pika::get<1>(t).get(), 42);
}

void test_wait_for_either_of_three_futures_3()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::future<int> f3(pt3.get_future());

    pt3();

    pika::future<pika::when_any_result<
        pika::tuple<pika::future<int>, pika::future<int>, pika::future<int>>>>
        r = pika::when_any(f1, f2, f3);
    pika::tuple<pika::future<int>, pika::future<int>, pika::future<int>> t =
        r.get().futures;

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());
    PIKA_TEST(!f3.valid());

    PIKA_TEST(pika::get<2>(t).is_ready());
    PIKA_TEST_EQ(pika::get<2>(t).get(), 42);
}

void test_wait_for_either_of_four_futures_1()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::future<int> f3(pt3.get_future());
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::future<int> f4(pt4.get_future());

    pt1();

    pika::future<pika::when_any_result<pika::tuple<pika::future<int>,
        pika::future<int>, pika::future<int>, pika::future<int>>>>
        r = pika::when_any(f1, f2, f3, f4);
    pika::tuple<pika::future<int>, pika::future<int>, pika::future<int>,
        pika::future<int>>
        t = r.get().futures;

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());
    PIKA_TEST(!f3.valid());
    PIKA_TEST(!f4.valid());

    PIKA_TEST(pika::get<0>(t).is_ready());
    PIKA_TEST_EQ(pika::get<0>(t).get(), 42);
}

void test_wait_for_either_of_four_futures_2()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::future<int> f3(pt3.get_future());
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::future<int> f4(pt4.get_future());

    pt2();

    pika::future<pika::when_any_result<pika::tuple<pika::future<int>,
        pika::future<int>, pika::future<int>, pika::future<int>>>>
        r = pika::when_any(f1, f2, f3, f4);
    pika::tuple<pika::future<int>, pika::future<int>, pika::future<int>,
        pika::future<int>>
        t = r.get().futures;

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());
    PIKA_TEST(!f3.valid());
    PIKA_TEST(!f4.valid());

    PIKA_TEST(pika::get<1>(t).is_ready());
    PIKA_TEST_EQ(pika::get<1>(t).get(), 42);
}

void test_wait_for_either_of_four_futures_3()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::future<int> f3(pt3.get_future());
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::future<int> f4(pt4.get_future());

    pt3();

    pika::future<pika::when_any_result<pika::tuple<pika::future<int>,
        pika::future<int>, pika::future<int>, pika::future<int>>>>
        r = pika::when_any(f1, f2, f3, f4);
    pika::tuple<pika::future<int>, pika::future<int>, pika::future<int>,
        pika::future<int>>
        t = r.get().futures;

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());
    PIKA_TEST(!f3.valid());
    PIKA_TEST(!f4.valid());

    PIKA_TEST(pika::get<2>(t).is_ready());
    PIKA_TEST_EQ(pika::get<2>(t).get(), 42);
}

void test_wait_for_either_of_four_futures_4()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::future<int> f3(pt3.get_future());
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::future<int> f4(pt4.get_future());

    pt4();

    pika::future<pika::when_any_result<pika::tuple<pika::future<int>,
        pika::future<int>, pika::future<int>, pika::future<int>>>>
        r = pika::when_any(f1, f2, f3, f4);
    pika::tuple<pika::future<int>, pika::future<int>, pika::future<int>,
        pika::future<int>>
        t = r.get().futures;

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());
    PIKA_TEST(!f3.valid());
    PIKA_TEST(!f4.valid());

    PIKA_TEST(pika::get<3>(t).is_ready());
    PIKA_TEST_EQ(pika::get<3>(t).get(), 42);
}

template <class Container>
void test_wait_for_either_of_five_futures_1_from_list()
{
    Container futures;

    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::future<int> f1(pt1.get_future());
    futures.push_back(std::move(f1));
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::future<int> f2(pt2.get_future());
    futures.push_back(std::move(f2));
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::future<int> f3(pt3.get_future());
    futures.push_back(std::move(f3));
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::future<int> f4(pt4.get_future());
    futures.push_back(std::move(f4));
    pika::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    pika::future<int> f5(pt5.get_future());
    futures.push_back(std::move(f5));

    pt1();

    pika::future<pika::when_any_result<Container>> r = pika::when_any(futures);
    pika::when_any_result<Container> raw = r.get();

    PIKA_TEST_EQ(raw.index, 0u);

    Container t = std::move(raw.futures);

    PIKA_TEST(!f1.valid());    // NOLINT
    PIKA_TEST(!f2.valid());    // NOLINT
    PIKA_TEST(!f3.valid());    // NOLINT
    PIKA_TEST(!f4.valid());    // NOLINT
    PIKA_TEST(!f5.valid());    // NOLINT

    PIKA_TEST(t.front().is_ready());
    PIKA_TEST_EQ(t.front().get(), 42);
}

template <class Container>
void test_wait_for_either_of_five_futures_1_from_list_iterators()
{
    typedef typename Container::iterator iterator;

    Container futures;

    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::future<int> f1(pt1.get_future());
    futures.push_back(std::move(f1));
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::future<int> f2(pt2.get_future());
    futures.push_back(std::move(f2));
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::future<int> f3(pt3.get_future());
    futures.push_back(std::move(f3));
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::future<int> f4(pt4.get_future());
    futures.push_back(std::move(f4));
    pika::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    pika::future<int> f5(pt5.get_future());
    futures.push_back(std::move(f5));

    pt1();

    pika::future<pika::when_any_result<Container>> r =
        pika::when_any<iterator, Container>(futures.begin(), futures.end());
    pika::when_any_result<Container> raw = r.get();

    PIKA_TEST_EQ(raw.index, 0u);

    Container t = std::move(raw.futures);

    PIKA_TEST(!f1.valid());    // NOLINT
    PIKA_TEST(!f2.valid());    // NOLINT
    PIKA_TEST(!f3.valid());    // NOLINT
    PIKA_TEST(!f4.valid());    // NOLINT
    PIKA_TEST(!f5.valid());    // NOLINT

    PIKA_TEST(t.front().is_ready());
    PIKA_TEST_EQ(t.front().get(), 42);
}

void test_wait_for_either_of_five_futures_1()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::future<int> f3(pt3.get_future());
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::future<int> f4(pt4.get_future());
    pika::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    pika::future<int> f5(pt5.get_future());

    pt1();

    pika::future<
        pika::when_any_result<pika::tuple<pika::future<int>, pika::future<int>,
            pika::future<int>, pika::future<int>, pika::future<int>>>>
        r = pika::when_any(f1, f2, f3, f4, f5);
    pika::tuple<pika::future<int>, pika::future<int>, pika::future<int>,
        pika::future<int>, pika::future<int>>
        t = r.get().futures;

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());
    PIKA_TEST(!f3.valid());
    PIKA_TEST(!f4.valid());
    PIKA_TEST(!f5.valid());

    PIKA_TEST(pika::get<0>(t).is_ready());
    PIKA_TEST_EQ(pika::get<0>(t).get(), 42);
}

void test_wait_for_either_of_five_futures_2()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::future<int> f3(pt3.get_future());
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::future<int> f4(pt4.get_future());
    pika::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    pika::future<int> f5(pt5.get_future());

    pt2();

    pika::future<
        pika::when_any_result<pika::tuple<pika::future<int>, pika::future<int>,
            pika::future<int>, pika::future<int>, pika::future<int>>>>
        r = pika::when_any(f1, f2, f3, f4, f5);
    pika::tuple<pika::future<int>, pika::future<int>, pika::future<int>,
        pika::future<int>, pika::future<int>>
        t = r.get().futures;

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());
    PIKA_TEST(!f3.valid());
    PIKA_TEST(!f4.valid());
    PIKA_TEST(!f5.valid());

    PIKA_TEST(pika::get<1>(t).is_ready());
    PIKA_TEST_EQ(pika::get<1>(t).get(), 42);
}

void test_wait_for_either_of_five_futures_3()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::future<int> f3(pt3.get_future());
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::future<int> f4(pt4.get_future());
    pika::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    pika::future<int> f5(pt5.get_future());

    pt3();

    pika::future<
        pika::when_any_result<pika::tuple<pika::future<int>, pika::future<int>,
            pika::future<int>, pika::future<int>, pika::future<int>>>>
        r = pika::when_any(f1, f2, f3, f4, f5);
    pika::tuple<pika::future<int>, pika::future<int>, pika::future<int>,
        pika::future<int>, pika::future<int>>
        t = r.get().futures;

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());
    PIKA_TEST(!f3.valid());
    PIKA_TEST(!f4.valid());
    PIKA_TEST(!f5.valid());

    PIKA_TEST(pika::get<2>(t).is_ready());
    PIKA_TEST_EQ(pika::get<2>(t).get(), 42);
}

void test_wait_for_either_of_five_futures_4()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::future<int> f3(pt3.get_future());
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::future<int> f4(pt4.get_future());
    pika::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    pika::future<int> f5(pt5.get_future());

    pt4();

    pika::future<
        pika::when_any_result<pika::tuple<pika::future<int>, pika::future<int>,
            pika::future<int>, pika::future<int>, pika::future<int>>>>
        r = pika::when_any(f1, f2, f3, f4, f5);
    pika::tuple<pika::future<int>, pika::future<int>, pika::future<int>,
        pika::future<int>, pika::future<int>>
        t = r.get().futures;

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());
    PIKA_TEST(!f3.valid());
    PIKA_TEST(!f4.valid());
    PIKA_TEST(!f5.valid());

    PIKA_TEST(pika::get<3>(t).is_ready());
    PIKA_TEST_EQ(pika::get<3>(t).get(), 42);
}

void test_wait_for_either_of_five_futures_5()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::future<int> f3(pt3.get_future());
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::future<int> f4(pt4.get_future());
    pika::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    pika::future<int> f5(pt5.get_future());

    pt5();

    pika::future<
        pika::when_any_result<pika::tuple<pika::future<int>, pika::future<int>,
            pika::future<int>, pika::future<int>, pika::future<int>>>>
        r = pika::when_any(f1, f2, f3, f4, f5);
    pika::tuple<pika::future<int>, pika::future<int>, pika::future<int>,
        pika::future<int>, pika::future<int>>
        t = r.get().futures;

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());
    PIKA_TEST(!f3.valid());
    PIKA_TEST(!f4.valid());
    PIKA_TEST(!f5.valid());

    PIKA_TEST(pika::get<4>(t).is_ready());
    PIKA_TEST_EQ(pika::get<4>(t).get(), 42);
}

///////////////////////////////////////////////////////////////////////////////
// void test_wait_for_either_invokes_callbacks()
// {
//     callback_called = 0;
//     pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
//     pika::future<int> fi = pt1.get_future();
//     pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
//     pika::future<int> fi2 = pt2.get_future();
//     pt1.set_wait_callback(wait_callback_for_task);
//
//     pika::thread t(std::move(pt));
//
//     boost::wait_for_any(fi, fi2);
//     PIKA_TEST_EQ(callback_called, 1U);
//     PIKA_TEST_EQ(fi.get(), 42);
// }

// void test_wait_for_any_from_range()
// {
//     unsigned const count = 10;
//     for(unsigned i = 0; i < count; ++i)
//     {
//         pika::lcos::local::packaged_task<int()> tasks[count];
//         pika::future<int> futures[count];
//         for(unsigned j = 0; j < count; ++j)
//         {
//             tasks[j] =
//               std::move(pika::lcos::local::packaged_task<int()>(make_int_slowly));
//             futures[j] = tasks[j].get_future();
//         }
//         pika::thread t(std::move(tasks[i]));
//
//         pika::wait_any(futures, futures);
//
//         pika::future<int>* const future =
//               boost::wait_for_any(futures, futures+count);
//
//         PIKA_TEST_EQ(future, (futures + i));
//         for(unsigned j = 0; j < count; ++j)
//         {
//             if (j != i)
//             {
//                 PIKA_TEST(!futures[j].is_ready());
//             }
//             else
//             {
//                 PIKA_TEST(futures[j].is_ready());
//             }
//         }
//         PIKA_TEST_EQ(futures[i].get(), 42);
//     }
// }

void test_wait_for_either_of_two_late_futures()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::future<int> f2(pt2.get_future());

    pika::future<
        pika::when_any_result<pika::tuple<pika::future<int>, pika::future<int>>>>
        r = pika::when_any(f1, f2);

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());

    pt2();
    pt1();

    pika::tuple<pika::future<int>, pika::future<int>> t = r.get().futures;

    PIKA_TEST(pika::get<1>(t).is_ready());
    PIKA_TEST_EQ(pika::get<1>(t).get(), 42);
}

void test_wait_for_either_of_two_deferred_futures()
{
    pika::future<int> f1 = pika::async(pika::launch::deferred, &make_int_slowly);
    pika::future<int> f2 = pika::async(pika::launch::deferred, &make_int_slowly);

    pika::future<
        pika::when_any_result<pika::tuple<pika::future<int>, pika::future<int>>>>
        r = pika::when_any(f1, f2);

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());

    pika::tuple<pika::future<int>, pika::future<int>> t = r.get().futures;

    PIKA_TEST(pika::get<0>(t).is_ready());
    PIKA_TEST_EQ(pika::get<0>(t).get(), 42);
}

///////////////////////////////////////////////////////////////////////////////
using pika::program_options::options_description;
using pika::program_options::variables_map;

using pika::future;

int pika_main(variables_map&)
{
    {
        test_wait_for_either_of_two_futures_1();
        test_wait_for_either_of_two_futures_2();
        test_wait_for_either_of_two_futures_list_1<std::vector<future<int>>>();
        test_wait_for_either_of_two_futures_list_1<std::list<future<int>>>();
        test_wait_for_either_of_two_futures_list_1<std::deque<future<int>>>();
        test_wait_for_either_of_two_futures_list_2<std::vector<future<int>>>();
        test_wait_for_either_of_two_futures_list_2<std::list<future<int>>>();
        test_wait_for_either_of_two_futures_list_2<std::deque<future<int>>>();
        test_wait_for_either_of_three_futures_1();
        test_wait_for_either_of_three_futures_2();
        test_wait_for_either_of_three_futures_3();
        test_wait_for_either_of_four_futures_1();
        test_wait_for_either_of_four_futures_2();
        test_wait_for_either_of_four_futures_3();
        test_wait_for_either_of_four_futures_4();
        test_wait_for_either_of_five_futures_1_from_list<
            std::vector<future<int>>>();
        test_wait_for_either_of_five_futures_1_from_list<
            std::list<future<int>>>();
        test_wait_for_either_of_five_futures_1_from_list<
            std::deque<future<int>>>();
        test_wait_for_either_of_five_futures_1_from_list_iterators<
            std::vector<future<int>>>();
        test_wait_for_either_of_five_futures_1_from_list_iterators<
            std::list<future<int>>>();
        test_wait_for_either_of_five_futures_1_from_list_iterators<
            std::deque<future<int>>>();
        test_wait_for_either_of_five_futures_1();
        test_wait_for_either_of_five_futures_2();
        test_wait_for_either_of_five_futures_3();
        test_wait_for_either_of_five_futures_4();
        test_wait_for_either_of_five_futures_5();
        //         test_wait_for_either_invokes_callbacks();
        //         test_wait_for_any_from_range();
        test_wait_for_either_of_two_late_futures();
        test_wait_for_either_of_two_deferred_futures();
    }

    pika::local::finalize();
    return pika::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " PIKA_APPLICATION_STRING " [options]");

    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.desc_cmdline = cmdline;
    init_args.cfg = cfg;

    return pika::local::init(pika_main, argc, argv, init_args);
}
