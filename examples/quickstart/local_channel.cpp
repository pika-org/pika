//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example demonstrates the use of a channel which is very similar to the
// equally named feature in the Go language.

#include <pika/local/channel.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>

#include <iostream>
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

    std::cout << "sum: " << x + y << std::endl;
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

    std::cout << "ping-ponged: " << pongs.get(pika::launch::sync) << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
void pingpong1()
{
    pika::lcos::local::one_element_channel<std::string> pings;
    pika::lcos::local::one_element_channel<std::string> pongs;

    for (int i = 0; i != 10; ++i)
    {
        ping(pings, "passed message");
        pong(pings, pongs);
        pongs.get(pika::launch::sync);
    }

    std::cout << "ping-ponged 10 times\n";
}

void pingpong2()
{
    pika::lcos::local::one_element_channel<std::string> pings;
    pika::lcos::local::one_element_channel<std::string> pongs;

    ping(pings, "passed message");
    pika::future<void> f1 = pika::async([=]() { pong(pings, pongs); });

    ping(pings, "passed message");
    pika::future<void> f2 = pika::async([=]() { pong(pings, pongs); });

    pongs.get(pika::launch::sync);
    pongs.get(pika::launch::sync);

    f1.get();
    f2.get();

    std::cout << "ping-ponged with waiting\n";
}

///////////////////////////////////////////////////////////////////////////////
void dispatch_work()
{
    pika::lcos::local::channel<int> jobs;
    pika::lcos::local::channel<> done;

    pika::apply([jobs, done]() mutable {
        while (true)
        {
            pika::error_code ec(pika::lightweight);
            int value = jobs.get(pika::launch::sync, ec);
            if (!ec)
            {
                std::cout << "received job: " << value << std::endl;
            }
            else
            {
                std::cout << "received all jobs" << std::endl;
                done.set();
                break;
            }
        }
    });

    for (int j = 1; j <= 3; ++j)
    {
        jobs.set(j);
        std::cout << "sent job: " << j << std::endl;
    }

    jobs.close();
    std::cout << "sent all jobs" << std::endl;

    done.get(pika::launch::sync);
}

///////////////////////////////////////////////////////////////////////////////
void channel_range()
{
    pika::lcos::local::channel<std::string> queue;

    queue.set("one");
    queue.set("two");
    queue.close();

    for (auto const& elem : queue)
        std::cout << elem << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    calculate_sum();
    pingpong();
    pingpong1();
    pingpong2();
    dispatch_work();
    channel_range();

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    return pika::local::init(pika_main, argc, argv);
}
