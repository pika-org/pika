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
#include <thread>
#include <memory>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int make_int_slowly()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return 42;
}

template <class Container>
void test_wait_for_all_from_list()
{
    unsigned const count = 10;
    Container futures;
    for (unsigned j = 0; j < count; ++j)
    {
        pika::lcos::local::futures_factory<int()> task(make_int_slowly);
        futures.push_back(task.get_future());
        task.apply();
    }

    pika::future<Container> r = pika::when_all(futures);

    Container result = r.get();

    PIKA_TEST_EQ(futures.size(), result.size());
    for (const auto& f : futures)
        PIKA_TEST(!f.valid());
    for (const auto& r : result)
        PIKA_TEST(r.is_ready());
}

template <class Container>
void test_wait_for_all_from_list_iterators()
{
    unsigned const count = 10;

    Container futures;
    for (unsigned j = 0; j < count; ++j)
    {
        pika::lcos::local::futures_factory<int()> task(make_int_slowly);
        futures.push_back(task.get_future());
        task.apply();
    }

    pika::future<Container> r =
        pika::when_all<typename Container::iterator, Container>(
            futures.begin(), futures.end());

    Container result = r.get();

    PIKA_TEST_EQ(futures.size(), result.size());
    for (const auto& f : futures)
        PIKA_TEST(!f.valid());
    for (const auto& r : result)
        PIKA_TEST(r.is_ready());
}

void test_wait_for_all_one_future()
{
    pika::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    pika::future<int> f1 = pt1.get_future();
    pt1.apply();

    typedef pika::tuple<pika::future<int>> result_type;
    pika::future<result_type> r = pika::when_all(f1);

    result_type result = r.get();

    PIKA_TEST(!f1.valid());

    PIKA_TEST(pika::get<0>(result).is_ready());
}

void test_wait_for_all_two_futures()
{
    pika::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    pika::future<int> f1 = pt1.get_future();
    pt1.apply();
    pika::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    pika::future<int> f2 = pt2.get_future();
    pt2.apply();

    typedef pika::tuple<pika::future<int>, pika::future<int>> result_type;
    pika::future<result_type> r = pika::when_all(f1, f2);

    result_type result = r.get();

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());

    PIKA_TEST(pika::get<0>(result).is_ready());
    PIKA_TEST(pika::get<1>(result).is_ready());
}

void test_wait_for_all_three_futures()
{
    pika::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    pika::future<int> f1 = pt1.get_future();
    pt1.apply();
    pika::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    pika::future<int> f2 = pt2.get_future();
    pt2.apply();
    pika::lcos::local::futures_factory<int()> pt3(make_int_slowly);
    pika::future<int> f3 = pt3.get_future();
    pt3.apply();

    typedef pika::tuple<pika::future<int>, pika::future<int>, pika::future<int>>
        result_type;
    pika::future<result_type> r = pika::when_all(f1, f2, f3);

    result_type result = r.get();

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());
    PIKA_TEST(!f3.valid());

    PIKA_TEST(pika::get<0>(result).is_ready());
    PIKA_TEST(pika::get<1>(result).is_ready());
    PIKA_TEST(pika::get<2>(result).is_ready());
}

void test_wait_for_all_four_futures()
{
    pika::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    pika::future<int> f1 = pt1.get_future();
    pt1.apply();
    pika::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    pika::future<int> f2 = pt2.get_future();
    pt2.apply();
    pika::lcos::local::futures_factory<int()> pt3(make_int_slowly);
    pika::future<int> f3 = pt3.get_future();
    pt3.apply();
    pika::lcos::local::futures_factory<int()> pt4(make_int_slowly);
    pika::future<int> f4 = pt4.get_future();
    pt4.apply();

    typedef pika::tuple<pika::future<int>, pika::future<int>, pika::future<int>,
        pika::future<int>>
        result_type;
    pika::future<result_type> r = pika::when_all(f1, f2, f3, f4);

    result_type result = r.get();

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());
    PIKA_TEST(!f3.valid());
    PIKA_TEST(!f4.valid());

    PIKA_TEST(pika::get<0>(result).is_ready());
    PIKA_TEST(pika::get<1>(result).is_ready());
    PIKA_TEST(pika::get<2>(result).is_ready());
    PIKA_TEST(pika::get<3>(result).is_ready());
}

void test_wait_for_all_five_futures()
{
    pika::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    pika::future<int> f1 = pt1.get_future();
    pt1.apply();
    pika::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    pika::future<int> f2 = pt2.get_future();
    pt2.apply();
    pika::lcos::local::futures_factory<int()> pt3(make_int_slowly);
    pika::future<int> f3 = pt3.get_future();
    pt3.apply();
    pika::lcos::local::futures_factory<int()> pt4(make_int_slowly);
    pika::future<int> f4 = pt4.get_future();
    pt4.apply();
    pika::lcos::local::futures_factory<int()> pt5(make_int_slowly);
    pika::future<int> f5 = pt5.get_future();
    pt5.apply();

    typedef pika::tuple<pika::future<int>, pika::future<int>, pika::future<int>,
        pika::future<int>, pika::future<int>>
        result_type;
    pika::future<result_type> r = pika::when_all(f1, f2, f3, f4, f5);

    result_type result = r.get();

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());
    PIKA_TEST(!f3.valid());
    PIKA_TEST(!f4.valid());
    PIKA_TEST(!f5.valid());

    PIKA_TEST(pika::get<0>(result).is_ready());
    PIKA_TEST(pika::get<1>(result).is_ready());
    PIKA_TEST(pika::get<2>(result).is_ready());
    PIKA_TEST(pika::get<3>(result).is_ready());
    PIKA_TEST(pika::get<4>(result).is_ready());
}

void test_wait_for_all_late_futures()
{
    pika::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    pika::future<int> f1 = pt1.get_future();
    pt1.apply();
    pika::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    pika::future<int> f2 = pt2.get_future();

    typedef pika::tuple<pika::future<int>, pika::future<int>> result_type;
    pika::future<result_type> r = pika::when_all(f1, f2);

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());

    pt2.apply();

    result_type result = r.get();

    PIKA_TEST(pika::get<0>(result).is_ready());
    PIKA_TEST(pika::get<1>(result).is_ready());
}

void test_wait_for_all_deferred_futures()
{
    pika::future<int> f1 = pika::async(pika::launch::deferred, &make_int_slowly);
    pika::future<int> f2 = pika::async(pika::launch::deferred, &make_int_slowly);

    typedef pika::tuple<pika::future<int>, pika::future<int>> result_type;
    pika::future<result_type> r = pika::when_all(f1, f2);

    PIKA_TEST(!f1.valid());
    PIKA_TEST(!f2.valid());

    result_type result = r.get();

    PIKA_TEST(pika::get<0>(result).is_ready());
    PIKA_TEST(pika::get<1>(result).is_ready());
}

///////////////////////////////////////////////////////////////////////////////
using pika::program_options::options_description;
using pika::program_options::variables_map;

using pika::future;

int pika_main(variables_map&)
{
    {
        test_wait_for_all_from_list<std::vector<future<int>>>();
        test_wait_for_all_from_list<std::list<future<int>>>();
        test_wait_for_all_from_list<std::deque<future<int>>>();
        test_wait_for_all_from_list_iterators<std::vector<future<int>>>();
        test_wait_for_all_from_list_iterators<std::list<future<int>>>();
        test_wait_for_all_from_list_iterators<std::deque<future<int>>>();
        test_wait_for_all_one_future();
        test_wait_for_all_two_futures();
        test_wait_for_all_three_futures();
        test_wait_for_all_four_futures();
        test_wait_for_all_five_futures();
        test_wait_for_all_late_futures();
        test_wait_for_all_deferred_futures();
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
