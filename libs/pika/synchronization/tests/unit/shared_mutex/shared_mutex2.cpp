// (C) Copyright 2006-7 Anthony Williams
//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/shared_mutex.hpp>
#include <pika/local/thread.hpp>

#include <pika/modules/testing.hpp>

#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "shared_mutex_locking_thread.hpp"
#include "thread_group.hpp"

#define CHECK_LOCKED_VALUE_EQUAL(mutex_name, value, expected_value)            \
    {                                                                          \
        std::unique_lock<pika::lcos::local::mutex> lock(mutex_name);            \
        PIKA_TEST_EQ(value, expected_value);                                    \
    }

void test_only_one_upgrade_lock_permitted()
{
    typedef pika::lcos::local::shared_mutex shared_mutex_type;
    typedef pika::lcos::local::mutex mutex_type;

    unsigned const number_of_threads = 2;

    test::thread_group pool;

    shared_mutex_type rw_mutex;
    unsigned unblocked_count = 0;
    unsigned simultaneous_running_count = 0;
    unsigned max_simultaneous_running = 0;
    mutex_type unblocked_count_mutex;
    pika::lcos::local::condition_variable unblocked_condition;
    mutex_type finish_mutex;
    std::unique_lock<mutex_type> finish_lock(finish_mutex);

    try
    {
        for (unsigned i = 0; i != number_of_threads; ++i)
        {
            pool.create_thread(
                test::locking_thread<pika::upgrade_lock<shared_mutex_type>>(
                    rw_mutex, unblocked_count, unblocked_count_mutex,
                    unblocked_condition, finish_mutex,
                    simultaneous_running_count, max_simultaneous_running));
        }

        pika::this_thread::yield();

        CHECK_LOCKED_VALUE_EQUAL(unblocked_count_mutex, unblocked_count, 1u);

        finish_lock.unlock();
        pool.join_all();
    }
    catch (...)
    {
        pool.interrupt_all();
        pool.join_all();
        PIKA_TEST(false);
    }

    CHECK_LOCKED_VALUE_EQUAL(
        unblocked_count_mutex, unblocked_count, number_of_threads);
    CHECK_LOCKED_VALUE_EQUAL(
        unblocked_count_mutex, max_simultaneous_running, 1u);
}

void test_can_lock_upgrade_if_currently_locked_shared()
{
    typedef pika::lcos::local::shared_mutex shared_mutex_type;
    typedef pika::lcos::local::mutex mutex_type;

    test::thread_group pool;

    shared_mutex_type rw_mutex;
    unsigned unblocked_count = 0;
    unsigned simultaneous_running_count = 0;
    unsigned max_simultaneous_running = 0;
    mutex_type unblocked_count_mutex;
    pika::lcos::local::condition_variable unblocked_condition;
    mutex_type finish_mutex;
    std::unique_lock<mutex_type> finish_lock(finish_mutex);

    unsigned const reader_count = 10;

    try
    {
        for (unsigned i = 0; i != reader_count; ++i)
        {
            pool.create_thread(
                test::locking_thread<std::shared_lock<shared_mutex_type>>(
                    rw_mutex, unblocked_count, unblocked_count_mutex,
                    unblocked_condition, finish_mutex,
                    simultaneous_running_count, max_simultaneous_running));
        }

        pika::this_thread::yield();

        pool.create_thread(
            test::locking_thread<pika::upgrade_lock<shared_mutex_type>>(rw_mutex,
                unblocked_count, unblocked_count_mutex, unblocked_condition,
                finish_mutex, simultaneous_running_count,
                max_simultaneous_running));

        {
            std::unique_lock<mutex_type> lk(unblocked_count_mutex);
            // NOLINTNEXTLINE(bugprone-infinite-loop)
            while (unblocked_count < (reader_count + 1))
            {
                unblocked_condition.wait(lk);
            }
        }

        CHECK_LOCKED_VALUE_EQUAL(
            unblocked_count_mutex, unblocked_count, reader_count + 1);

        finish_lock.unlock();
        pool.join_all();
    }
    catch (...)
    {
        pool.interrupt_all();
        pool.join_all();
        PIKA_TEST(false);
    }

    CHECK_LOCKED_VALUE_EQUAL(
        unblocked_count_mutex, unblocked_count, reader_count + 1);
    CHECK_LOCKED_VALUE_EQUAL(
        unblocked_count_mutex, max_simultaneous_running, reader_count + 1);
}

void test_can_lock_upgrade_to_unique_if_currently_locked_upgrade()
{
    typedef pika::lcos::local::shared_mutex shared_mutex_type;

    shared_mutex_type mtx;
    pika::upgrade_lock<shared_mutex_type> l(mtx);
    pika::upgrade_to_unique_lock<shared_mutex_type> ul(l);
    PIKA_TEST(ul.owns_lock());
}

void test_if_other_thread_has_write_lock_try_lock_shared_returns_false()
{
    typedef pika::lcos::local::shared_mutex shared_mutex_type;
    typedef pika::lcos::local::mutex mutex_type;

    shared_mutex_type rw_mutex;
    mutex_type finish_mutex;
    mutex_type unblocked_mutex;
    unsigned unblocked_count = 0;
    std::unique_lock<mutex_type> finish_lock(finish_mutex);
    pika::thread writer(test::simple_writing_thread(
        rw_mutex, finish_mutex, unblocked_mutex, unblocked_count));

    std::this_thread::sleep_for(std::chrono::seconds(1));

    CHECK_LOCKED_VALUE_EQUAL(unblocked_mutex, unblocked_count, 1u);

    bool const try_succeeded = rw_mutex.try_lock_shared();
    PIKA_TEST(!try_succeeded);
    if (try_succeeded)
    {
        rw_mutex.unlock_shared();
    }

    finish_lock.unlock();
    writer.join();
}

void test_if_other_thread_has_write_lock_try_lock_upgrade_returns_false()
{
    typedef pika::lcos::local::shared_mutex shared_mutex_type;
    typedef pika::lcos::local::mutex mutex_type;

    shared_mutex_type rw_mutex;
    mutex_type finish_mutex;
    mutex_type unblocked_mutex;
    unsigned unblocked_count = 0;
    std::unique_lock<mutex_type> finish_lock(finish_mutex);
    pika::thread writer(test::simple_writing_thread(
        rw_mutex, finish_mutex, unblocked_mutex, unblocked_count));

    std::this_thread::sleep_for(std::chrono::seconds(1));

    CHECK_LOCKED_VALUE_EQUAL(unblocked_mutex, unblocked_count, 1u);

    bool const try_succeeded = rw_mutex.try_lock_upgrade();
    PIKA_TEST(!try_succeeded);
    if (try_succeeded)
    {
        rw_mutex.unlock_upgrade();
    }

    finish_lock.unlock();
    writer.join();
}

void test_if_no_thread_has_lock_try_lock_shared_returns_true()
{
    typedef pika::lcos::local::shared_mutex shared_mutex_type;

    shared_mutex_type rw_mutex;
    bool const try_succeeded = rw_mutex.try_lock_shared();
    PIKA_TEST(try_succeeded);
    if (try_succeeded)
    {
        rw_mutex.unlock_shared();
    }
}

void test_if_no_thread_has_lock_try_lock_upgrade_returns_true()
{
    typedef pika::lcos::local::shared_mutex shared_mutex_type;

    shared_mutex_type rw_mutex;
    bool const try_succeeded = rw_mutex.try_lock_upgrade();
    PIKA_TEST(try_succeeded);
    if (try_succeeded)
    {
        rw_mutex.unlock_upgrade();
    }
}

void test_if_other_thread_has_shared_lock_try_lock_shared_returns_true()
{
    typedef pika::lcos::local::shared_mutex shared_mutex_type;
    typedef pika::lcos::local::mutex mutex_type;

    shared_mutex_type rw_mutex;
    mutex_type finish_mutex;
    mutex_type unblocked_mutex;
    unsigned unblocked_count = 0;
    std::unique_lock<mutex_type> finish_lock(finish_mutex);
    pika::thread writer(test::simple_reading_thread(
        rw_mutex, finish_mutex, unblocked_mutex, unblocked_count));

    std::this_thread::sleep_for(std::chrono::seconds(1));

    CHECK_LOCKED_VALUE_EQUAL(unblocked_mutex, unblocked_count, 1u);

    bool const try_succeeded = rw_mutex.try_lock_shared();
    PIKA_TEST(try_succeeded);
    if (try_succeeded)
    {
        rw_mutex.unlock_shared();
    }

    finish_lock.unlock();
    writer.join();
}

void test_if_other_thread_has_shared_lock_try_lock_upgrade_returns_true()
{
    typedef pika::lcos::local::shared_mutex shared_mutex_type;
    typedef pika::lcos::local::mutex mutex_type;

    shared_mutex_type rw_mutex;
    mutex_type finish_mutex;
    mutex_type unblocked_mutex;
    unsigned unblocked_count = 0;
    std::unique_lock<mutex_type> finish_lock(finish_mutex);
    pika::thread writer(test::simple_reading_thread(
        rw_mutex, finish_mutex, unblocked_mutex, unblocked_count));

    std::this_thread::sleep_for(std::chrono::seconds(1));

    CHECK_LOCKED_VALUE_EQUAL(unblocked_mutex, unblocked_count, 1u);

    bool const try_succeeded = rw_mutex.try_lock_upgrade();
    PIKA_TEST(try_succeeded);
    if (try_succeeded)
    {
        rw_mutex.unlock_upgrade();
    }

    finish_lock.unlock();
    writer.join();
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    test_only_one_upgrade_lock_permitted();
    test_can_lock_upgrade_if_currently_locked_shared();
    test_can_lock_upgrade_to_unique_if_currently_locked_upgrade();
    test_if_other_thread_has_write_lock_try_lock_shared_returns_false();
    test_if_other_thread_has_write_lock_try_lock_upgrade_returns_false();
    test_if_no_thread_has_lock_try_lock_shared_returns_true();
    test_if_no_thread_has_lock_try_lock_upgrade_returns_true();
    test_if_other_thread_has_shared_lock_try_lock_shared_returns_true();
    test_if_other_thread_has_shared_lock_try_lock_upgrade_returns_true();

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.cfg = cfg;
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv, init_args), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
