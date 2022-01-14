//  Taken from the Boost.Thread library
//
// Copyright (C) 2001-2003 William E. Kempf
// Copyright (C) 2007-2008 Anthony Williams
// Copyright (C) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/modules/threading.hpp>
#include <pika/synchronization/condition_variable.hpp>
#include <pika/synchronization/mutex.hpp>
#include <pika/topology/topology.hpp>

#include <chrono>
#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace {
    pika::lcos::local::mutex multiple_wake_mutex;
    pika::lcos::local::condition_variable multiple_wake_cond;
    unsigned multiple_wake_count = 0;

    void wait_for_condvar_and_increase_count()
    {
        std::unique_lock<pika::lcos::local::mutex> lk(multiple_wake_mutex);
        multiple_wake_cond.wait(lk);
        ++multiple_wake_count;
    }

    void join_all(std::vector<pika::thread>& group)
    {
        for (std::size_t i = 0; i < group.size(); ++i)
            group[i].join();
    }

}    // namespace

///////////////////////////////////////////////////////////////////////////////
struct wait_for_flag
{
    pika::lcos::local::mutex mutex;
    pika::lcos::local::condition_variable cond_var;
    bool flag;
    unsigned woken;

    wait_for_flag()
      : flag(false)
      , woken(0)
    {
    }

    struct check_flag
    {
        bool const& flag;

        check_flag(bool const& flag_)
          : flag(flag_)
        {
        }

        bool operator()() const
        {
            return flag;
        }
    };

    void wait_without_predicate()
    {
        std::unique_lock<pika::lcos::local::mutex> lock(mutex);
        while (!flag)
        {
            cond_var.wait(lock);
        }
        ++woken;
    }

    void wait_with_predicate()
    {
        std::unique_lock<pika::lcos::local::mutex> lock(mutex);
        cond_var.wait(lock, check_flag(flag));
        if (flag)
        {
            ++woken;
        }
    }

    void wait_until_without_predicate()
    {
        std::chrono::system_clock::time_point const timeout =
            std::chrono::system_clock::now() + std::chrono::milliseconds(5);

        std::unique_lock<pika::lcos::local::mutex> lock(mutex);
        while (!flag)
        {
            if (cond_var.wait_until(lock, timeout) ==
                pika::lcos::local::cv_status::timeout)
            {
                return;
            }
        }
        ++woken;
    }

    void wait_until_with_predicate()
    {
        std::chrono::system_clock::time_point const timeout =
            std::chrono::system_clock::now() + std::chrono::milliseconds(5);

        std::unique_lock<pika::lcos::local::mutex> lock(mutex);
        if (cond_var.wait_until(lock, timeout, check_flag(flag)) && flag)
        {
            ++woken;
        }
    }
    void relative_wait_until_with_predicate()
    {
        std::unique_lock<pika::lcos::local::mutex> lock(mutex);
        if (cond_var.wait_for(
                lock, std::chrono::milliseconds(5), check_flag(flag)) &&
            flag)
        {
            ++woken;
        }
    }
};

void test_condition_notify_one_wakes_from_wait()
{
    wait_for_flag data;

    pika::thread thread(&wait_for_flag::wait_without_predicate, std::ref(data));

    {
        std::unique_lock<pika::lcos::local::mutex> lock(data.mutex);
        data.flag = true;
    }

    data.cond_var.notify_one();

    thread.join();
    PIKA_TEST(data.woken);
}

void test_condition_notify_one_wakes_from_wait_with_predicate()
{
    wait_for_flag data;

    pika::thread thread(&wait_for_flag::wait_with_predicate, std::ref(data));

    {
        std::unique_lock<pika::lcos::local::mutex> lock(data.mutex);
        data.flag = true;
    }

    data.cond_var.notify_one();

    thread.join();
    PIKA_TEST(data.woken);
}

void test_condition_notify_one_wakes_from_wait_until()
{
    wait_for_flag data;

    pika::thread thread(
        &wait_for_flag::wait_until_without_predicate, std::ref(data));

    {
        std::unique_lock<pika::lcos::local::mutex> lock(data.mutex);
        data.flag = true;
    }

    data.cond_var.notify_one();

    thread.join();
    PIKA_TEST(data.woken);
}

void test_condition_notify_one_wakes_from_wait_until_with_predicate()
{
    wait_for_flag data;

    pika::thread thread(
        &wait_for_flag::wait_until_with_predicate, std::ref(data));

    {
        std::unique_lock<pika::lcos::local::mutex> lock(data.mutex);
        data.flag = true;
    }

    data.cond_var.notify_one();

    thread.join();
    PIKA_TEST(data.woken);
}

void test_condition_notify_one_wakes_from_relative_wait_until_with_predicate()
{
    wait_for_flag data;

    pika::thread thread(
        &wait_for_flag::relative_wait_until_with_predicate, std::ref(data));

    {
        std::unique_lock<pika::lcos::local::mutex> lock(data.mutex);
        data.flag = true;
    }

    data.cond_var.notify_one();

    thread.join();
    PIKA_TEST(data.woken);
}

void test_multiple_notify_one_calls_wakes_multiple_threads()
{
    multiple_wake_count = 0;

    pika::thread thread1(wait_for_condvar_and_increase_count);
    pika::thread thread2(wait_for_condvar_and_increase_count);

    pika::this_thread::yield();
    multiple_wake_cond.notify_one();

    pika::thread thread3(wait_for_condvar_and_increase_count);

    pika::this_thread::yield();
    multiple_wake_cond.notify_one();
    multiple_wake_cond.notify_one();
    pika::this_thread::yield();

    {
        std::unique_lock<pika::lcos::local::mutex> lk(multiple_wake_mutex);
        PIKA_TEST(multiple_wake_count == 3);
    }

    thread1.join();
    thread2.join();
    thread3.join();
}

///////////////////////////////////////////////////////////////////////////////

void test_condition_notify_all_wakes_from_wait()
{
    wait_for_flag data;

    std::vector<pika::thread> group;

    try
    {
        for (unsigned i = 0; i < 5; ++i)
        {
            group.push_back(pika::thread(
                &wait_for_flag::wait_without_predicate, std::ref(data)));
        }

        {
            std::unique_lock<pika::lcos::local::mutex> lock(data.mutex);
            data.flag = true;
        }

        data.cond_var.notify_all();

        join_all(group);
        PIKA_TEST_EQ(data.woken, 5u);
    }
    catch (...)
    {
        join_all(group);
        throw;
    }
}

void test_condition_notify_all_wakes_from_wait_with_predicate()
{
    wait_for_flag data;

    std::vector<pika::thread> group;

    try
    {
        for (unsigned i = 0; i < 5; ++i)
        {
            group.push_back(pika::thread(
                &wait_for_flag::wait_with_predicate, std::ref(data)));
        }

        {
            std::unique_lock<pika::lcos::local::mutex> lock(data.mutex);
            data.flag = true;
        }

        data.cond_var.notify_all();

        join_all(group);
        PIKA_TEST_EQ(data.woken, 5u);
    }
    catch (...)
    {
        join_all(group);
        throw;
    }
}

void test_condition_notify_all_wakes_from_wait_until()
{
    wait_for_flag data;

    std::vector<pika::thread> group;

    try
    {
        for (unsigned i = 0; i < 5; ++i)
        {
            group.push_back(pika::thread(
                &wait_for_flag::wait_until_without_predicate, std::ref(data)));
        }

        {
            std::unique_lock<pika::lcos::local::mutex> lock(data.mutex);
            data.flag = true;
        }

        data.cond_var.notify_all();

        join_all(group);
        PIKA_TEST_EQ(data.woken, 5u);
    }
    catch (...)
    {
        join_all(group);
        throw;
    }
}

void test_condition_notify_all_wakes_from_wait_until_with_predicate()
{
    wait_for_flag data;

    std::vector<pika::thread> group;

    try
    {
        for (unsigned i = 0; i < 5; ++i)
        {
            group.push_back(pika::thread(
                &wait_for_flag::wait_until_with_predicate, std::ref(data)));
        }

        {
            std::unique_lock<pika::lcos::local::mutex> lock(data.mutex);
            data.flag = true;
        }

        data.cond_var.notify_all();

        join_all(group);
        PIKA_TEST_EQ(data.woken, 5u);
    }
    catch (...)
    {
        join_all(group);
        throw;
    }
}

void test_condition_notify_all_wakes_from_relative_wait_until_with_predicate()
{
    wait_for_flag data;

    std::vector<pika::thread> group;

    try
    {
        for (unsigned i = 0; i < 5; ++i)
        {
            group.push_back(
                pika::thread(&wait_for_flag ::relative_wait_until_with_predicate,
                    std::ref(data)));
        }

        {
            std::unique_lock<pika::lcos::local::mutex> lock(data.mutex);
            data.flag = true;
        }

        data.cond_var.notify_all();

        join_all(group);
        PIKA_TEST_EQ(data.woken, 5u);
    }
    catch (...)
    {
        join_all(group);
        throw;
    }
}

void test_notify_all_following_notify_one_wakes_all_threads()
{
    multiple_wake_count = 0;

    pika::thread thread1(wait_for_condvar_and_increase_count);
    pika::thread thread2(wait_for_condvar_and_increase_count);

    pika::this_thread::yield();
    multiple_wake_cond.notify_one();

    pika::thread thread3(wait_for_condvar_and_increase_count);

    pika::this_thread::yield();
    multiple_wake_cond.notify_one();
    multiple_wake_cond.notify_all();
    pika::this_thread::yield();

    {
        std::unique_lock<pika::lcos::local::mutex> lk(multiple_wake_mutex);
        PIKA_TEST(multiple_wake_count == 3);
    }

    thread1.join();
    thread2.join();
    thread3.join();
}

///////////////////////////////////////////////////////////////////////////////
struct condition_test_data
{
    condition_test_data()
      : notified(0)
      , awoken(0)
    {
    }

    pika::lcos::local::mutex mutex;
    pika::lcos::local::condition_variable condition;
    int notified;
    int awoken;
};

void condition_test_thread(condition_test_data* data)
{
    std::unique_lock<pika::lcos::local::mutex> lock(data->mutex);
    PIKA_TEST(lock ? true : false);
    while (!(data->notified > 0))
        data->condition.wait(lock);
    PIKA_TEST(lock ? true : false);
    data->awoken++;
}

struct cond_predicate
{
    cond_predicate(int& var, int val)
      : _var(var)
      , _val(val)
    {
    }

    bool operator()()
    {
        return _var == _val;
    }

    int& _var;
    int _val;
};

void condition_test_waits(condition_test_data* data)
{
    std::unique_lock<pika::lcos::local::mutex> lock(data->mutex);
    PIKA_TEST(lock ? true : false);

    // Test wait.
    while (data->notified != 1)
        data->condition.wait(lock);
    PIKA_TEST(lock ? true : false);
    PIKA_TEST_EQ(data->notified, 1);
    data->awoken++;
    data->condition.notify_one();

    // Test predicate wait.
    data->condition.wait(lock, cond_predicate(data->notified, 2));
    PIKA_TEST(lock ? true : false);
    PIKA_TEST_EQ(data->notified, 2);
    data->awoken++;
    data->condition.notify_one();

    // Test wait_until.
    std::chrono::system_clock::time_point xt =
        std::chrono::system_clock::now() + std::chrono::milliseconds(100);
    while (data->notified != 3)
        data->condition.wait_until(lock, xt);
    PIKA_TEST(lock ? true : false);
    PIKA_TEST_EQ(data->notified, 3);
    data->awoken++;
    data->condition.notify_one();

    // Test predicate wait_until.
    xt = std::chrono::system_clock::now() + std::chrono::milliseconds(100);
    cond_predicate pred(data->notified, 4);
    PIKA_TEST(data->condition.wait_until(lock, xt, pred));
    PIKA_TEST(lock ? true : false);
    PIKA_TEST(pred());
    PIKA_TEST_EQ(data->notified, 4);
    data->awoken++;
    data->condition.notify_one();

    // Test predicate wait_for
    cond_predicate pred_rel(data->notified, 5);
    PIKA_TEST(data->condition.wait_for(
        lock, std::chrono::milliseconds(100), pred_rel));
    PIKA_TEST(lock ? true : false);
    PIKA_TEST(pred_rel());
    PIKA_TEST_EQ(data->notified, 5);
    data->awoken++;
    data->condition.notify_one();
}

void test_condition_waits()
{
    typedef std::unique_lock<pika::lcos::local::mutex> unique_lock;

    condition_test_data data;

    pika::thread thread(&condition_test_waits, &data);

    {
        unique_lock lock(data.mutex);
        PIKA_TEST(lock ? true : false);

        {
            pika::util::unlock_guard<unique_lock> ul(lock);
            pika::this_thread::yield();
        }

        data.notified++;
        data.condition.notify_one();
        while (data.awoken != 1)
            data.condition.wait(lock);
        PIKA_TEST(lock ? true : false);
        PIKA_TEST_EQ(data.awoken, 1);

        {
            pika::util::unlock_guard<unique_lock> ul(lock);
            pika::this_thread::yield();
        }

        data.notified++;
        data.condition.notify_one();
        while (data.awoken != 2)
            data.condition.wait(lock);
        PIKA_TEST(lock ? true : false);
        PIKA_TEST_EQ(data.awoken, 2);

        {
            pika::util::unlock_guard<unique_lock> ul(lock);
            pika::this_thread::yield();
        }

        data.notified++;
        data.condition.notify_one();
        while (data.awoken != 3)
            data.condition.wait(lock);
        PIKA_TEST(lock ? true : false);
        PIKA_TEST_EQ(data.awoken, 3);

        {
            pika::util::unlock_guard<unique_lock> ul(lock);
            pika::this_thread::yield();
        }

        data.notified++;
        data.condition.notify_one();
        while (data.awoken != 4)
            data.condition.wait(lock);
        PIKA_TEST(lock ? true : false);
        PIKA_TEST_EQ(data.awoken, 4);

        {
            pika::util::unlock_guard<unique_lock> ul(lock);
            pika::this_thread::yield();
        }

        data.notified++;
        data.condition.notify_one();
        while (data.awoken != 5)
            data.condition.wait(lock);
        PIKA_TEST(lock ? true : false);
        PIKA_TEST_EQ(data.awoken, 5);
    }

    thread.join();
    PIKA_TEST_EQ(data.awoken, 5);
}

///////////////////////////////////////////////////////////////////////////////

bool fake_predicate()
{
    return false;
}

std::chrono::milliseconds const delay(1000);
std::chrono::milliseconds const timeout_resolution(100);

void test_wait_until_times_out()
{
    pika::lcos::local::condition_variable cond;
    pika::lcos::local::mutex m;

    std::unique_lock<pika::lcos::local::mutex> lock(m);
    std::chrono::system_clock::time_point const start =
        std::chrono::system_clock::now();
    std::chrono::system_clock::time_point const timeout = start + delay;

    while (cond.wait_until(lock, timeout) ==
        pika::lcos::local::cv_status::no_timeout)
    {
    }

    std::chrono::system_clock::time_point const end =
        std::chrono::system_clock::now();
    PIKA_TEST_LTE((delay - timeout_resolution).count(), (end - start).count());
}

void test_wait_until_with_predicate_times_out()
{
    pika::lcos::local::condition_variable cond;
    pika::lcos::local::mutex m;

    std::unique_lock<pika::lcos::local::mutex> lock(m);
    std::chrono::system_clock::time_point const start =
        std::chrono::system_clock::now();
    std::chrono::system_clock::time_point const timeout = start + delay;

    bool const res = cond.wait_until(lock, timeout, fake_predicate);

    std::chrono::system_clock::time_point const end =
        std::chrono::system_clock::now();
    PIKA_TEST(!res);
    PIKA_TEST_LTE((delay - timeout_resolution).count(), (end - start).count());
}

void test_relative_wait_until_with_predicate_times_out()
{
    pika::lcos::local::condition_variable cond;
    pika::lcos::local::mutex m;

    std::unique_lock<pika::lcos::local::mutex> lock(m);
    std::chrono::system_clock::time_point const start =
        std::chrono::system_clock::now();

    bool const res = cond.wait_for(lock, delay, fake_predicate);

    std::chrono::system_clock::time_point const end =
        std::chrono::system_clock::now();
    PIKA_TEST(!res);
    PIKA_TEST_LTE((delay - timeout_resolution).count(), (end - start).count());
}

void test_wait_until_relative_times_out()
{
    pika::lcos::local::condition_variable cond;
    pika::lcos::local::mutex m;

    std::unique_lock<pika::lcos::local::mutex> lock(m);
    std::chrono::system_clock::time_point const start =
        std::chrono::system_clock::now();

    while (
        cond.wait_for(lock, delay) == pika::lcos::local::cv_status::no_timeout)
    {
    }

    std::chrono::system_clock::time_point const end =
        std::chrono::system_clock::now();
    PIKA_TEST_LTE((delay - timeout_resolution).count(), (end - start).count());
}

///////////////////////////////////////////////////////////////////////////////
using pika::program_options::options_description;
using pika::program_options::variables_map;

int pika_main(variables_map&)
{
    {
        test_condition_notify_one_wakes_from_wait();
        test_condition_notify_one_wakes_from_wait_with_predicate();
        test_condition_notify_one_wakes_from_wait_until();
        test_condition_notify_one_wakes_from_wait_until_with_predicate();
        test_condition_notify_one_wakes_from_relative_wait_until_with_predicate();
        test_multiple_notify_one_calls_wakes_multiple_threads();
    }
    {
        test_condition_notify_all_wakes_from_wait();
        test_condition_notify_all_wakes_from_wait_with_predicate();
        test_condition_notify_all_wakes_from_wait_until();
        test_condition_notify_all_wakes_from_wait_until_with_predicate();
        test_condition_notify_all_wakes_from_relative_wait_until_with_predicate();
        test_notify_all_following_notify_one_wakes_all_threads();
    }
    {
        test_condition_waits();
    }
    {
        test_wait_until_times_out();
        test_wait_until_with_predicate_times_out();
        test_relative_wait_until_with_predicate_times_out();
        test_wait_until_relative_times_out();
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
