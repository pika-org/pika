//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <atomic>
#include <numeric>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
void sum(std::vector<int> const& s, pika::lcos::local::channel<int> c)
{
    c.set(std::accumulate(s.begin(), s.end(), 0));    // send sum to channel
}

void calculate_sum()
{
    std::vector<int> s = {7, 2, 8, -9, 4, 0};
    pika::lcos::local::channel<int> c;

    pika::apply(&sum, std::vector<int>(s.begin(), s.begin() + s.size() / 2), c);
    pika::apply(&sum, std::vector<int>(s.begin() + s.size() / 2, s.end()), c);

    int x = c.get(pika::launch::sync);    // receive from c
    int y = c.get(pika::launch::sync);

    int expected = std::accumulate(s.begin(), s.end(), 0);
    PIKA_TEST_EQ(expected, x + y);
}

///////////////////////////////////////////////////////////////////////////////
void ping(
    pika::lcos::local::send_channel<std::string> pings, std::string const& msg)
{
    pings.set(msg);
}

void pong(pika::lcos::local::receive_channel<std::string> pings,
    pika::lcos::local::send_channel<std::string> pongs)
{
    std::string msg = pings.get(pika::launch::sync);
    pongs.set(msg);
}

void pingpong()
{
    pika::lcos::local::channel<std::string> pings;
    pika::lcos::local::channel<std::string> pongs;

    ping(pings, "passed message");
    pong(pings, pongs);

    std::string result = pongs.get(pika::launch::sync);
    PIKA_TEST_EQ(std::string("passed message"), result);
}

void pingpong1()
{
    pika::lcos::local::one_element_channel<std::string> pings;
    pika::lcos::local::one_element_channel<std::string> pongs;

    for (int i = 0; i != 10; ++i)
    {
        ping(pings, "passed message");
        pong(pings, pongs);

        std::string result = pongs.get(pika::launch::sync);
        PIKA_TEST_EQ(std::string("passed message"), result);
    }
}

///////////////////////////////////////////////////////////////////////////////
void ping_void(pika::lcos::local::send_channel<> pings)
{
    pings.set();
}

void pong_void(pika::lcos::local::receive_channel<> pings,
    pika::lcos::local::send_channel<> pongs, bool& pingponged)
{
    pings.get(pika::launch::sync);
    pongs.set();

    PIKA_TEST(!pingponged);
    pingponged = true;
}

void pingpong_void()
{
    pika::lcos::local::channel<> pings;
    pika::lcos::local::channel<> pongs;

    bool pingponged = false;

    ping_void(pings);
    pong_void(pings, pongs, pingponged);

    pongs.get(pika::launch::sync);
    PIKA_TEST(pingponged);
}

void pingpong_void1()
{
    pika::lcos::local::one_element_channel<> pings;
    pika::lcos::local::one_element_channel<> pongs;

    for (int i = 0; i != 10; ++i)
    {
        bool pingponged = false;

        ping_void(pings);
        pong_void(pings, pongs, pingponged);

        pongs.get(pika::launch::sync);
        PIKA_TEST(pingponged);
    }
}

///////////////////////////////////////////////////////////////////////////////
void dispatch_work()
{
    pika::lcos::local::channel<int> jobs;
    pika::lcos::local::channel<> done;

    std::atomic<int> received_jobs(0);
    std::atomic<bool> was_closed(false);

    pika::apply([jobs, done, &received_jobs, &was_closed]() mutable {
        while (true)
        {
            pika::error_code ec(pika::lightweight);
            int next = jobs.get(pika::launch::sync, ec);
            (void) next;
            if (!ec)
            {
                ++received_jobs;
            }
            else
            {
                was_closed = true;
                done.set();
                break;
            }
        }
    });

    for (int j = 1; j <= 3; ++j)
    {
        jobs.set(j);
    }

    jobs.close();
    done.get(pika::launch::sync);

    PIKA_TEST_EQ(received_jobs.load(), 3);
    PIKA_TEST(was_closed.load());
}

///////////////////////////////////////////////////////////////////////////////
void channel_range()
{
    std::atomic<int> received_elements(0);

    pika::lcos::local::channel<std::string> queue;
    queue.set("one");
    queue.set("two");
    queue.set("three");
    queue.close();

    for (auto const& elem : queue)
    {
        (void) elem;
        ++received_elements;
    }

    PIKA_TEST_EQ(received_elements.load(), 3);
}

void channel_range_void()
{
    std::atomic<int> received_elements(0);

    pika::lcos::local::channel<> queue;
    queue.set();
    queue.set();
    queue.set();
    queue.close();

    for (auto const& elem : queue)
    {
        (void) elem;
        ++received_elements;
    }

    PIKA_TEST_EQ(received_elements.load(), 3);
}

///////////////////////////////////////////////////////////////////////////////
void deadlock_test()
{
    bool caught_exception = false;
    try
    {
        pika::lcos::local::channel<int> c;
        int value = c.get(pika::launch::sync);
        PIKA_TEST(false);
        (void) value;
    }
    catch (pika::exception const&)
    {
        caught_exception = true;
    }
    PIKA_TEST(caught_exception);
}

void closed_channel_get()
{
    bool caught_exception = false;
    try
    {
        pika::lcos::local::channel<int> c;
        c.close();

        int value = c.get(pika::launch::sync);
        PIKA_TEST(false);
        (void) value;
    }
    catch (pika::exception const&)
    {
        caught_exception = true;
    }
    PIKA_TEST(caught_exception);
}

void closed_channel_get_generation()
{
    bool caught_exception = false;
    try
    {
        pika::lcos::local::channel<int> c;
        c.set(42, 122);    // setting value for generation 122
        c.close();

        PIKA_TEST_EQ(c.get(pika::launch::sync, 122), 42);

        int value =
            c.get(pika::launch::sync, 123);    // asking for generation 123
        PIKA_TEST(false);
        (void) value;
    }
    catch (pika::exception const&)
    {
        caught_exception = true;
    }
    PIKA_TEST(caught_exception);
}

void closed_channel_set()
{
    bool caught_exception = false;
    try
    {
        pika::lcos::local::channel<int> c;
        c.close();

        c.set(42);
        PIKA_TEST(false);
    }
    catch (pika::exception const&)
    {
        caught_exception = true;
    }
    PIKA_TEST(caught_exception);
}

///////////////////////////////////////////////////////////////////////////////
void deadlock_test1()
{
    bool caught_exception = false;
    try
    {
        pika::lcos::local::one_element_channel<int> c;
        int value = c.get(pika::launch::sync);
        PIKA_TEST(false);
        (void) value;
    }
    catch (pika::exception const&)
    {
        caught_exception = true;
    }
    PIKA_TEST(caught_exception);
}

void closed_channel_get1()
{
    bool caught_exception = false;
    try
    {
        pika::lcos::local::one_element_channel<int> c;
        c.close();

        int value = c.get(pika::launch::sync);
        PIKA_TEST(false);
        (void) value;
    }
    catch (pika::exception const&)
    {
        caught_exception = true;
    }
    PIKA_TEST(caught_exception);
}

void closed_channel_set1()
{
    bool caught_exception = false;
    try
    {
        pika::lcos::local::one_element_channel<int> c;
        c.close();

        c.set(42);
        PIKA_TEST(false);
    }
    catch (pika::exception const&)
    {
        caught_exception = true;
    }
    PIKA_TEST(caught_exception);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    calculate_sum();
    pingpong();
    pingpong1();
    pingpong_void();
    pingpong_void1();
    dispatch_work();
    channel_range();
    channel_range_void();

    deadlock_test();
    closed_channel_get();
    closed_channel_get_generation();
    closed_channel_set();

    deadlock_test1();
    closed_channel_get1();
    closed_channel_set1();

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
