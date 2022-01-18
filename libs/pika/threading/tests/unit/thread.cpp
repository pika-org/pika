// Copyright (C) 2012 Hartmut Kaiser
// Copyright (C) 2001-2003 William E. Kempf
// Copyright (C) 2008 Anthony Williams
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
inline void set_description(char const* test_name)
{
    pika::threads::set_thread_description(
        pika::threads::get_self_id(), test_name);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Clock, typename Duration>
inline int time_cmp(std::chrono::time_point<Clock, Duration> const& xt1,
    std::chrono::time_point<Clock, Duration> const& xt2)
{
    if (xt1 == xt2)
        return 0;
    return xt1 > xt2 ? 1 : -1;
}

template <typename Clock, typename Duration, typename Rep, typename Period>
inline bool in_range(std::chrono::time_point<Clock, Duration> const& xt,
    std::chrono::duration<Rep, Period> const& d)
{
    std::chrono::time_point<Clock, Duration> const now = Clock::now();
    std::chrono::time_point<Clock, Duration> const mint = now - d;
    return time_cmp(xt, mint) >= 0 && time_cmp(xt, now) <= 0;
}

///////////////////////////////////////////////////////////////////////////////
template <typename F>
void timed_test(F func, int /*secs*/)
{
    pika::thread thrd(func);
    thrd.join();

    // FIXME: implement execution monitor to verify in-time execution and to
    //        prevent deadlocks
}

///////////////////////////////////////////////////////////////////////////////
int test_value = 0;

void simple_thread()
{
    test_value = 999;
}

void comparison_thread(pika::thread::id parent)
{
    pika::thread::id const my_id = pika::this_thread::get_id();
    PIKA_TEST_NEQ(my_id, parent);

    pika::thread::id const my_id2 = pika::this_thread::get_id();
    PIKA_TEST_EQ(my_id, my_id2);

    pika::thread::id const no_thread_id = pika::thread::id();
    PIKA_TEST_NEQ(my_id, no_thread_id);
}

///////////////////////////////////////////////////////////////////////////////
void test_sleep()
{
    set_description("test_sleep");

    std::chrono::system_clock::time_point const now =
        std::chrono::system_clock::now();
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // Ensure it's in a range instead of checking actual equality due to time
    // lapse
    PIKA_TEST(in_range(now, std::chrono::seconds(4)));    //-V112
}

///////////////////////////////////////////////////////////////////////////////
void do_test_creation()
{
    test_value = 0;
    pika::thread thrd(&simple_thread);
    thrd.join();
    PIKA_TEST_EQ(test_value, 999);
}

void test_creation()
{
    set_description("test_creation");
    timed_test(&do_test_creation, 1);
}

///////////////////////////////////////////////////////////////////////////////
void do_test_id_comparison()
{
    pika::thread::id const self = pika::this_thread::get_id();
    pika::thread thrd(&comparison_thread, self);
    thrd.join();
}

void test_id_comparison()
{
    set_description("test_id_comparison");
    timed_test(&do_test_id_comparison, 1);
}

///////////////////////////////////////////////////////////////////////////////
void interruption_point_thread(pika::lcos::local::spinlock* m, bool* failed)
{
    std::unique_lock<pika::lcos::local::spinlock> lk(*m);
    pika::util::ignore_while_checking il(&lk);
    PIKA_UNUSED(il);

    pika::this_thread::interruption_point();
    *failed = true;
}

void do_test_thread_interrupts_at_interruption_point()
{
    pika::lcos::local::spinlock m;
    bool failed = false;
    std::unique_lock<pika::lcos::local::spinlock> lk(m);
    pika::thread thrd(&interruption_point_thread, &m, &failed);
    thrd.interrupt();
    lk.unlock();
    thrd.join();
    PIKA_TEST(!failed);
}

void test_thread_interrupts_at_interruption_point()
{
    set_description("test_thread_interrupts_at_interruption_point");
    timed_test(&do_test_thread_interrupts_at_interruption_point, 1);
}

///////////////////////////////////////////////////////////////////////////////
void disabled_interruption_point_thread(
    pika::lcos::local::spinlock* m, pika::lcos::local::barrier* b, bool* failed)
{
    pika::this_thread::disable_interruption dc;
    b->wait();
    try
    {
        std::lock_guard<pika::lcos::local::spinlock> lk(*m);
        pika::this_thread::interruption_point();
        *failed = false;
    }
    catch (...)
    {
        b->wait();
        throw;
    }
    b->wait();
}

void do_test_thread_no_interrupt_if_interrupts_disabled_at_interruption_point()
{
    pika::lcos::local::spinlock m;
    pika::lcos::local::barrier b(2);
    bool caught = false;
    bool failed = true;
    pika::thread thrd(&disabled_interruption_point_thread, &m, &b, &failed);
    b.wait();    // Make sure the test thread has been started and marked itself
                 // to disable interrupts.
    try
    {
        std::unique_lock<pika::lcos::local::spinlock> lk(m);
        pika::util::ignore_while_checking il(&lk);
        PIKA_UNUSED(il);

        thrd.interrupt();
    }
    catch (pika::exception& e)
    {
        PIKA_TEST_EQ(e.get_error(), pika::thread_not_interruptable);
        caught = true;
    }

    b.wait();

    thrd.join();
    PIKA_TEST(!failed);
    PIKA_TEST(caught);
}

void test_thread_no_interrupt_if_interrupts_disabled_at_interruption_point()
{
    set_description("test_thread_no_interrupt_if_interrupts_disabled_at\
                    _interruption_point");
    timed_test(
        &do_test_thread_no_interrupt_if_interrupts_disabled_at_interruption_point,
        1);
}

///////////////////////////////////////////////////////////////////////////////
struct non_copyable_callable
{
    unsigned value;

    non_copyable_callable()
      : value(0)
    {
    }

    non_copyable_callable(non_copyable_callable const&) = delete;
    non_copyable_callable& operator=(non_copyable_callable const&) = delete;

    void operator()()
    {
        value = 999;
    }
};

void do_test_creation_through_reference_wrapper()
{
    non_copyable_callable f;

    pika::thread thrd(std::ref(f));
    thrd.join();
    PIKA_TEST_EQ(f.value, 999u);
}

void test_creation_through_reference_wrapper()
{
    set_description("test_creation_through_reference_wrapper");
    timed_test(&do_test_creation_through_reference_wrapper, 1);
}

///////////////////////////////////////////////////////////////////////////////
// struct long_running_thread
// {
//     std::condition_variable cond;
//     std::mutex mut;
//     bool done;
//
//     long_running_thread()
//       : done(false)
//     {}
//
//     void operator()()
//     {
//         std::lock_guard<std::mutex> lk(mut);
//         while(!done)
//         {
//             cond.wait(lk);
//         }
//     }
// };
//
// void do_test_timed_join()
// {
//     long_running_thread f;
//     pika::thread thrd(std::ref(f));
//     PIKA_TEST(thrd.joinable());
//     std::chrono::system_clock::time_point xt =
//         std::chrono::system_clock::now()
//       + std::chrono::seconds(3);
//     bool const joined=thrd.timed_join(xt);
//     PIKA_TEST(in_range(xt, std::chrono::seconds(2)));
//     PIKA_TEST(!joined);
//     PIKA_TEST(thrd.joinable());
//     {
//         std::lock_guard<std::mutex> lk(f.mut);
//         f.done=true;
//         f.cond.notify_one();
//     }
//
//     xt = std::chrono::system_clock::now()
//       + std::chrono::seconds(3);
//     bool const joined2=thrd.timed_join(xt);
//     std::chrono::system_clock::time_point const now =
//         std::chrono::system_clock::now();
//     PIKA_TEST(xt>now);
//     PIKA_TEST(joined2);
//     PIKA_TEST(!thrd.joinable());
// }
//
// void test_timed_join()
// {
//     timed_test(&do_test_timed_join, 10);
// }

void simple_sync_thread(
    pika::lcos::local::barrier& b1, pika::lcos::local::barrier& b2)
{
    b1.wait();    // wait for both threads to be started
    // ... do nothing
    b2.wait();    // wait for the tests to be completed
}

void test_swap()
{
    set_description("test_swap");

    pika::lcos::local::barrier b1(3);
    pika::lcos::local::barrier b2(3);
    pika::thread t1(&simple_sync_thread, std::ref(b1), std::ref(b2));
    pika::thread t2(&simple_sync_thread, std::ref(b1), std::ref(b2));

    b1.wait();    // wait for both threads to be started

    pika::thread::id id1 = t1.get_id();
    pika::thread::id id2 = t2.get_id();

    t1.swap(t2);
    PIKA_TEST_EQ(t1.get_id(), id2);
    PIKA_TEST_EQ(t2.get_id(), id1);

    swap(t1, t2);
    PIKA_TEST_EQ(t1.get_id(), id1);
    PIKA_TEST_EQ(t2.get_id(), id2);

    b2.wait();    // wait for the tests to be completed

    t1.join();
    t2.join();
}

void test_double_join()
{
    set_description("test_double_join");

    pika::thread t([]() {});
    t.join();

    bool threw = true;
    bool caught = false;
    try
    {
        t.join();
        threw = false;
    }
    catch (pika::exception& e)
    {
        PIKA_TEST_EQ(e.get_error(), pika::invalid_status);
        caught = true;
    }

    PIKA_TEST(threw);
    PIKA_TEST(caught);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(variables_map&)
{
    {
        test_sleep();
        test_creation();
        test_id_comparison();
        test_thread_interrupts_at_interruption_point();
        test_thread_no_interrupt_if_interrupts_disabled_at_interruption_point();
        test_creation_through_reference_wrapper();
        test_swap();
        test_double_join();
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
