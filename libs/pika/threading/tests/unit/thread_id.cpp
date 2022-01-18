// Copyright (C) 2013 Hartmut Kaiser
// Copyright (C) 2007 Anthony Williams
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/barrier.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>

#include <chrono>
#include <functional>
#include <thread>

using pika::program_options::options_description;
using pika::program_options::variables_map;

///////////////////////////////////////////////////////////////////////////////
void do_nothing(pika::lcos::local::barrier& b1, pika::lcos::local::barrier& b2)
{
    b1.wait();
    std::this_thread::sleep_for(
        std::chrono::milliseconds(100));    // wait for 100 ms
    b2.wait();
}

void test_thread_id_for_default_constructed_thread_is_default_constructed_id()
{
    pika::thread t;
    PIKA_TEST_EQ(t.get_id(), pika::thread::id());
}

void test_thread_id_for_running_thread_is_not_default_constructed_id()
{
    pika::lcos::local::barrier b1(2);
    pika::lcos::local::barrier b2(2);
    pika::thread t(&do_nothing, std::ref(b1), std::ref(b2));
    b1.wait();

    PIKA_TEST_NEQ(t.get_id(), pika::thread::id());

    b2.wait();
    t.join();
}

void test_different_threads_have_different_ids()
{
    pika::lcos::local::barrier b1(3);
    pika::lcos::local::barrier b2(3);

    pika::thread t(&do_nothing, std::ref(b1), std::ref(b2));
    pika::thread t2(&do_nothing, std::ref(b1), std::ref(b2));
    b1.wait();

    PIKA_TEST_NEQ(t.get_id(), t2.get_id());

    b2.wait();
    t.join();
    t2.join();
}

void test_thread_ids_have_a_total_order()
{
    pika::lcos::local::barrier b1(4);
    pika::lcos::local::barrier b2(4);

    pika::thread t1(&do_nothing, std::ref(b1), std::ref(b2));
    pika::thread t2(&do_nothing, std::ref(b1), std::ref(b2));
    pika::thread t3(&do_nothing, std::ref(b1), std::ref(b2));
    b1.wait();

    pika::thread::id t1_id = t1.get_id();
    pika::thread::id t2_id = t2.get_id();
    pika::thread::id t3_id = t3.get_id();

    PIKA_TEST_NEQ(t1_id, t2_id);
    PIKA_TEST_NEQ(t1_id, t3_id);
    PIKA_TEST_NEQ(t2_id, t3_id);

    PIKA_TEST((t1_id < t2_id) != (t2_id < t1_id));
    PIKA_TEST((t1_id < t3_id) != (t3_id < t1_id));
    PIKA_TEST((t2_id < t3_id) != (t3_id < t2_id));

    PIKA_TEST((t1_id > t2_id) != (t2_id > t1_id));
    PIKA_TEST((t1_id > t3_id) != (t3_id > t1_id));
    PIKA_TEST((t2_id > t3_id) != (t3_id > t2_id));

    PIKA_TEST((t1_id < t2_id) == (t2_id > t1_id));
    PIKA_TEST((t2_id < t1_id) == (t1_id > t2_id));
    PIKA_TEST((t1_id < t3_id) == (t3_id > t1_id));
    PIKA_TEST((t3_id < t1_id) == (t1_id > t3_id));
    PIKA_TEST((t2_id < t3_id) == (t3_id > t2_id));
    PIKA_TEST((t3_id < t2_id) == (t2_id > t3_id));

    PIKA_TEST((t1_id < t2_id) == (t2_id >= t1_id));
    PIKA_TEST((t2_id < t1_id) == (t1_id >= t2_id));
    PIKA_TEST((t1_id < t3_id) == (t3_id >= t1_id));
    PIKA_TEST((t3_id < t1_id) == (t1_id >= t3_id));
    PIKA_TEST((t2_id < t3_id) == (t3_id >= t2_id));
    PIKA_TEST((t3_id < t2_id) == (t2_id >= t3_id));

    PIKA_TEST((t1_id <= t2_id) == (t2_id > t1_id));
    PIKA_TEST((t2_id <= t1_id) == (t1_id > t2_id));
    PIKA_TEST((t1_id <= t3_id) == (t3_id > t1_id));
    PIKA_TEST((t3_id <= t1_id) == (t1_id > t3_id));
    PIKA_TEST((t2_id <= t3_id) == (t3_id > t2_id));
    PIKA_TEST((t3_id <= t2_id) == (t2_id > t3_id));

    if ((t1_id < t2_id) && (t2_id < t3_id))
    {
        PIKA_TEST_LT(t1_id, t3_id);
    }
    else if ((t1_id < t3_id) && (t3_id < t2_id))
    {
        PIKA_TEST_LT(t1_id, t2_id);
    }
    else if ((t2_id < t3_id) && (t3_id < t1_id))
    {
        PIKA_TEST_LT(t2_id, t1_id);
    }
    else if ((t2_id < t1_id) && (t1_id < t3_id))
    {
        PIKA_TEST_LT(t2_id, t3_id);
    }
    else if ((t3_id < t1_id) && (t1_id < t2_id))
    {
        PIKA_TEST_LT(t3_id, t2_id);
    }
    else if ((t3_id < t2_id) && (t2_id < t1_id))
    {
        PIKA_TEST_LT(t3_id, t1_id);
    }
    else
    {
        PIKA_TEST(false);
    }

    pika::thread::id default_id;

    PIKA_TEST_LT(default_id, t1_id);
    PIKA_TEST_LT(default_id, t2_id);
    PIKA_TEST_LT(default_id, t3_id);

    PIKA_TEST_LTE(default_id, t1_id);
    PIKA_TEST_LTE(default_id, t2_id);
    PIKA_TEST_LTE(default_id, t3_id);

    PIKA_TEST(!(default_id > t1_id));
    PIKA_TEST(!(default_id > t2_id));
    PIKA_TEST(!(default_id > t3_id));

    PIKA_TEST(!(default_id >= t1_id));
    PIKA_TEST(!(default_id >= t2_id));
    PIKA_TEST(!(default_id >= t3_id));

    b2.wait();

    t1.join();
    t2.join();
    t3.join();
}

void get_thread_id(pika::thread::id* id)
{
    *id = pika::this_thread::get_id();
}

void test_thread_id_of_running_thread_returned_by_this_thread_get_id()
{
    pika::thread::id id;
    pika::thread t(&get_thread_id, &id);
    pika::thread::id t_id = t.get_id();
    t.join();
    PIKA_TEST_EQ(id, t_id);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(variables_map&)
{
    {
        test_thread_id_for_default_constructed_thread_is_default_constructed_id();
        test_thread_id_for_running_thread_is_not_default_constructed_id();
        test_different_threads_have_different_ids();
        test_thread_ids_have_a_total_order();
        test_thread_id_of_running_thread_returned_by_this_thread_get_id();
    }

    pika::local::finalize();
    return pika::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " PIKA_APPLICATION_STRING " [options]");

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return pika::local::init(pika_main, argc, argv, init_args);
}
