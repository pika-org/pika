//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>

#include <array>
#include <chrono>
#include <string>
#include <thread>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
pika::tuple<> make_tuple0_slowly()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return pika::make_tuple();
}

void test_split_future0()
{
    pika::lcos::local::futures_factory<pika::tuple<>()> pt(make_tuple0_slowly);
    pt.apply();

    pika::tuple<pika::future<void>> result =
        pika::split_future(pika::shared_future<pika::tuple<>>(pt.get_future()));

    pika::get<0>(result).get();
}

///////////////////////////////////////////////////////////////////////////////
pika::tuple<int> make_tuple1_slowly()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return pika::make_tuple(42);
}

void test_split_future1()
{
    pika::lcos::local::futures_factory<pika::tuple<int>()> pt(make_tuple1_slowly);
    pt.apply();

    pika::tuple<pika::future<int>> result =
        pika::split_future(pika::shared_future<pika::tuple<int>>(pt.get_future()));

    PIKA_TEST_EQ(pika::get<0>(result).get(), 42);
}

///////////////////////////////////////////////////////////////////////////////
pika::tuple<int, int> make_tuple2_slowly()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return pika::make_tuple(42, 43);
}

void test_split_future2()
{
    pika::lcos::local::futures_factory<pika::tuple<int, int>()> pt(
        make_tuple2_slowly);
    pt.apply();

    pika::tuple<pika::future<int>, pika::future<int>> result = pika::split_future(
        pika::shared_future<pika::tuple<int, int>>(pt.get_future()));

    PIKA_TEST_EQ(pika::get<0>(result).get(), 42);
    PIKA_TEST_EQ(pika::get<1>(result).get(), 43);
}

///////////////////////////////////////////////////////////////////////////////
pika::tuple<int, int, int> make_tuple3_slowly()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return pika::make_tuple(42, 43, 44);
}

void test_split_future3()
{
    pika::lcos::local::futures_factory<pika::tuple<int, int, int>()> pt(
        make_tuple3_slowly);
    pt.apply();

    pika::tuple<pika::future<int>, pika::future<int>, pika::future<int>> result =
        pika::split_future(
            pika::shared_future<pika::tuple<int, int, int>>(pt.get_future()));

    PIKA_TEST_EQ(pika::get<0>(result).get(), 42);
    PIKA_TEST_EQ(pika::get<1>(result).get(), 43);
    PIKA_TEST_EQ(pika::get<2>(result).get(), 44);
}

///////////////////////////////////////////////////////////////////////////////
std::pair<int, int> make_pair_slowly()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return std::make_pair(42, 43);
}

void test_split_future_pair()
{
    pika::lcos::local::futures_factory<std::pair<int, int>()> pt(
        make_pair_slowly);
    pt.apply();

    std::pair<pika::future<int>, pika::future<int>> result = pika::split_future(
        pika::shared_future<std::pair<int, int>>(pt.get_future()));

    PIKA_TEST_EQ(result.first.get(), 42);
    PIKA_TEST_EQ(result.second.get(), 43);
}

///////////////////////////////////////////////////////////////////////////////
std::array<int, 0> make_array0_slowly()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return std::array<int, 0>();
}

void test_split_future_array0()
{
    pika::lcos::local::futures_factory<std::array<int, 0>()> pt(
        make_array0_slowly);
    pt.apply();

    std::array<pika::future<void>, 1> result =
        pika::split_future(pt.get_future());

    result[0].get();
}

///////////////////////////////////////////////////////////////////////////////
std::array<int, 3> make_array_slowly()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return std::array<int, 3>{{42, 43, 44}};
}

void test_split_future_array()
{
    pika::lcos::local::futures_factory<std::array<int, 3>()> pt(
        make_array_slowly);
    pt.apply();

    std::array<pika::future<int>, 3> result = pika::split_future(pt.get_future());

    PIKA_TEST_EQ(result[0].get(), 42);
    PIKA_TEST_EQ(result[1].get(), 43);
    PIKA_TEST_EQ(result[2].get(), 44);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    test_split_future0();
    test_split_future1();
    test_split_future2();
    test_split_future3();

    test_split_future_pair();

    test_split_future_array0();
    test_split_future_array();

    pika::local::finalize();
    return pika::util::report_errors();
}

int main(int argc, char* argv[])
{
    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.cfg = cfg;

    return pika::local::init(pika_main, argc, argv, init_args);
}
