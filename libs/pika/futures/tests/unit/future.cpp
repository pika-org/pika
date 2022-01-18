//  Copyright (C) 2012-2017 Hartmut Kaiser
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
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct X
{
    int i;

    X()
      : i(42)
    {
    }

    X(X&& other)
      : i(other.i)
    {
        other.i = 0;
    }

    ~X() {}
};

int make_int()
{
    return 42;
}

int throw_runtime_error()
{
    throw std::runtime_error("42");
}

void set_promise_thread(pika::lcos::local::promise<int>* p)
{
    p->set_value(42);
}

struct my_exception
{
};

void set_promise_exception_thread(pika::lcos::local::promise<int>* p)
{
    p->set_exception(std::make_exception_ptr(my_exception()));
}

///////////////////////////////////////////////////////////////////////////////
void test_store_value_from_thread()
{
    pika::lcos::local::promise<int> pi2;
    pika::future<int> fi2(pi2.get_future());
    pika::thread t(&set_promise_thread, &pi2);
    fi2.wait();
    PIKA_TEST(fi2.is_ready());
    PIKA_TEST(fi2.has_value());
    PIKA_TEST(!fi2.has_exception());
    int j = fi2.get();
    PIKA_TEST_EQ(j, 42);
    t.join();
}

///////////////////////////////////////////////////////////////////////////////
void test_store_exception()
{
    pika::lcos::local::promise<int> pi3;
    pika::future<int> fi3 = pi3.get_future();
    pika::thread t(&set_promise_exception_thread, &pi3);
    fi3.wait();

    PIKA_TEST(fi3.is_ready());
    PIKA_TEST(!fi3.has_value());
    PIKA_TEST(fi3.has_exception());
    try
    {
        fi3.get();
        PIKA_TEST(false);
    }
    catch (my_exception)
    {
        PIKA_TEST(true);
    }
    t.join();
}

///////////////////////////////////////////////////////////////////////////////
void test_initial_state()
{
    pika::future<int> fi;
    PIKA_TEST(!fi.is_ready());
    PIKA_TEST(!fi.has_value());
    PIKA_TEST(!fi.has_exception());
    try
    {
        fi.get();
        PIKA_TEST(false);
    }
    catch (pika::exception const& e)
    {
        PIKA_TEST_EQ(e.get_error(), pika::no_state);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_waiting_future()
{
    pika::lcos::local::promise<int> pi;
    pika::future<int> fi;
    fi = pi.get_future();

    PIKA_TEST(!fi.is_ready());
    PIKA_TEST(!fi.has_value());
    PIKA_TEST(!fi.has_exception());

    // fulfill the promise so the destructor of promise is happy.
    pi.set_value(0);
}

///////////////////////////////////////////////////////////////////////////////
void test_cannot_get_future_twice()
{
    pika::lcos::local::promise<int> pi;
    pi.get_future();

    try
    {
        pi.get_future();
        PIKA_TEST(false);
    }
    catch (pika::exception const& e)
    {
        PIKA_TEST_EQ(e.get_error(), pika::future_already_retrieved);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_set_value_updates_future_status()
{
    pika::lcos::local::promise<int> pi;
    pika::future<int> fi;
    fi = pi.get_future();

    pi.set_value(42);

    PIKA_TEST(fi.is_ready());
    PIKA_TEST(fi.has_value());
    PIKA_TEST(!fi.has_exception());
}

///////////////////////////////////////////////////////////////////////////////
void test_set_value_can_be_retrieved()
{
    pika::lcos::local::promise<int> pi;
    pika::future<int> fi;
    fi = pi.get_future();

    pi.set_value(42);

    fi.wait();
    PIKA_TEST(fi.is_ready());
    PIKA_TEST(fi.has_value());
    PIKA_TEST(!fi.has_exception());
    int i = fi.get();
    PIKA_TEST_EQ(i, 42);
}

///////////////////////////////////////////////////////////////////////////////
void test_set_value_can_be_moved()
{
    pika::lcos::local::promise<int> pi;
    pika::future<int> fi;
    fi = pi.get_future();

    pi.set_value(42);

    fi.wait();
    PIKA_TEST(fi.is_ready());
    PIKA_TEST(fi.has_value());
    PIKA_TEST(!fi.has_exception());
    int i = 0;
    PIKA_TEST(i = fi.get());
    PIKA_TEST_EQ(i, 42);
}

///////////////////////////////////////////////////////////////////////////////
void test_future_from_packaged_task_is_waiting()
{
    pika::lcos::local::packaged_task<int()> pt(make_int);
    pika::future<int> fi = pt.get_future();

    PIKA_TEST(!fi.is_ready());
    PIKA_TEST(!fi.has_value());
    PIKA_TEST(!fi.has_exception());
}

///////////////////////////////////////////////////////////////////////////////
void test_invoking_a_packaged_task_populates_future()
{
    pika::lcos::local::packaged_task<int()> pt(make_int);
    pika::future<int> fi = pt.get_future();

    pt();

    PIKA_TEST(fi.is_ready());
    PIKA_TEST(fi.has_value());
    PIKA_TEST(!fi.has_exception());

    int i = fi.get();
    PIKA_TEST_EQ(i, 42);
}

///////////////////////////////////////////////////////////////////////////////
void test_invoking_a_packaged_task_twice_throws()
{
    pika::lcos::local::packaged_task<int()> pt(make_int);

    pt();
    try
    {
        pt();
        PIKA_TEST(false);
    }
    catch (pika::exception const& e)
    {
        PIKA_TEST_EQ(e.get_error(), pika::promise_already_satisfied);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }
    // retrieve the future so the destructor of packaged_task is happy.
    // Otherwise an exception will be tried to set to future_data which
    // leads to another exception to the fact that the future has already been
    // set.
    pt.get_future().get();
}

///////////////////////////////////////////////////////////////////////////////
void test_cannot_get_future_twice_from_task()
{
    pika::lcos::local::packaged_task<int()> pt(make_int);
    pt.get_future();
    try
    {
        pt.get_future();
        PIKA_TEST(false);
    }
    catch (pika::exception const& e)
    {
        PIKA_TEST_EQ(e.get_error(), pika::future_already_retrieved);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }
}

void test_task_stores_exception_if_function_throws()
{
    pika::lcos::local::packaged_task<int()> pt(throw_runtime_error);
    pika::future<int> fi = pt.get_future();

    pt();

    PIKA_TEST(fi.is_ready());
    PIKA_TEST(!fi.has_value());
    PIKA_TEST(fi.has_exception());
    try
    {
        fi.get();
        PIKA_TEST(false);
    }
    catch (std::exception&)
    {
        PIKA_TEST(true);
    }
    catch (...)
    {
        PIKA_TEST(!"Unknown exception thrown");
    }
}

void test_void_promise()
{
    pika::lcos::local::promise<void> p;
    pika::future<void> f = p.get_future();

    p.set_value();
    PIKA_TEST(f.is_ready());
    PIKA_TEST(f.has_value());
    PIKA_TEST(!f.has_exception());
}

void test_reference_promise()
{
    pika::lcos::local::promise<int&> p;
    pika::future<int&> f = p.get_future();
    int i = 42;
    p.set_value(i);
    PIKA_TEST(f.is_ready());
    PIKA_TEST(f.has_value());
    PIKA_TEST(!f.has_exception());
    PIKA_TEST_EQ(&f.get(), &i);
}

void do_nothing() {}

void test_task_returning_void()
{
    pika::lcos::local::packaged_task<void()> pt(do_nothing);
    pika::future<void> fi = pt.get_future();

    pt();

    PIKA_TEST(fi.is_ready());
    PIKA_TEST(fi.has_value());
    PIKA_TEST(!fi.has_exception());
}

int global_ref_target = 0;

int& return_ref()
{
    return global_ref_target;
}

void test_task_returning_reference()
{
    pika::lcos::local::packaged_task<int&()> pt(return_ref);
    pika::future<int&> fi = pt.get_future();

    pt();

    PIKA_TEST(fi.is_ready());
    PIKA_TEST(fi.has_value());
    PIKA_TEST(!fi.has_exception());
    int& i = fi.get();
    PIKA_TEST_EQ(&i, &global_ref_target);
}

void test_future_for_move_only_udt()
{
    pika::lcos::local::promise<X> pt;
    pika::future<X> fi = pt.get_future();

    pt.set_value(X());
    X res(fi.get());
    PIKA_TEST_EQ(res.i, 42);
}

void test_future_for_string()
{
    pika::lcos::local::promise<std::string> pt;
    pika::future<std::string> fi1 = pt.get_future();

    pt.set_value(std::string("hello"));
    std::string res(fi1.get());
    PIKA_TEST_EQ(res, "hello");

    pika::lcos::local::promise<std::string> pt2;
    fi1 = pt2.get_future();

    std::string const s = "goodbye";

    pt2.set_value(s);
    res = fi1.get();
    PIKA_TEST_EQ(res, "goodbye");

    pika::lcos::local::promise<std::string> pt3;
    fi1 = pt3.get_future();

    std::string s2 = "foo";

    pt3.set_value(s2);
    res = fi1.get();
    PIKA_TEST_EQ(res, "foo");
}

pika::lcos::local::spinlock callback_mutex;
unsigned callback_called = 0;

void wait_callback(pika::future<int>)
{
    std::lock_guard<pika::lcos::local::spinlock> lk(callback_mutex);
    ++callback_called;
}

void promise_set_value(pika::lcos::local::promise<int>& pi)
{
    try
    {
        pi.set_value(42);
    }
    catch (...)
    {
    }
}

void test_wait_callback()
{
    callback_called = 0;
    pika::lcos::local::promise<int> pi;
    pika::future<int> fi = pi.get_future();

    pika::future<void> ft = fi.then(&wait_callback);
    pika::thread t(&promise_set_value, std::ref(pi));

    ft.wait();

    t.join();

    PIKA_TEST_EQ(callback_called, 1U);
    ft.wait();
    ft.wait();
    ft.get();
    PIKA_TEST_EQ(callback_called, 1U);
}

void do_nothing_callback(pika::lcos::local::promise<int>& /*pi*/)
{
    std::lock_guard<pika::lcos::local::spinlock> lk(callback_mutex);
    ++callback_called;
}

void test_wait_callback_with_timed_wait()
{
    callback_called = 0;
    pika::lcos::local::promise<int> pi;
    pika::future<int> fi = pi.get_future();

    pika::future<void> fv =
        fi.then(pika::util::bind(&do_nothing_callback, std::ref(pi)));

    int state = int(fv.wait_for(std::chrono::milliseconds(100)));
    PIKA_TEST_EQ(state, int(pika::future_status::timeout));
    PIKA_TEST_EQ(callback_called, 0U);

    state = int(fv.wait_for(std::chrono::milliseconds(100)));
    PIKA_TEST_EQ(state, int(pika::future_status::timeout));
    state = int(fv.wait_for(std::chrono::milliseconds(100)));
    PIKA_TEST_EQ(state, int(pika::future_status::timeout));
    PIKA_TEST_EQ(callback_called, 0U);

    pi.set_value(42);

    state = int(fv.wait_for(std::chrono::milliseconds(100)));
    PIKA_TEST_EQ(state, int(pika::future_status::ready));

    PIKA_TEST_EQ(callback_called, 1U);
}

void test_packaged_task_can_be_moved()
{
    pika::lcos::local::packaged_task<int()> pt(make_int);
    pika::future<int> fi = pt.get_future();
    PIKA_TEST(!fi.is_ready());

    pika::lcos::local::packaged_task<int()> pt2(std::move(pt));
    PIKA_TEST(!fi.is_ready());

    try
    {
        pt();    // NOLINT
        PIKA_TEST(!"Can invoke moved task!");
    }
    catch (pika::exception const& e)
    {
        PIKA_TEST_EQ(e.get_error(), pika::no_state);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(!fi.is_ready());

    pt2();

    PIKA_TEST(fi.is_ready());
}

void test_destroying_a_promise_stores_broken_promise()
{
    pika::future<int> f;

    {
        pika::lcos::local::promise<int> p;
        f = p.get_future();
    }

    PIKA_TEST(f.is_ready());
    PIKA_TEST(f.has_exception());
    try
    {
        f.get();
        PIKA_TEST(false);    // shouldn't get here
    }
    catch (pika::exception const& e)
    {
        PIKA_TEST_EQ(e.get_error(), pika::broken_promise);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }
}

void test_destroying_a_packaged_task_stores_broken_task()
{
    pika::future<int> f;

    {
        pika::lcos::local::packaged_task<int()> p(make_int);
        f = p.get_future();
    }

    PIKA_TEST(f.is_ready());
    PIKA_TEST(f.has_exception());
    try
    {
        f.get();
        PIKA_TEST(false);    // shouldn't get here
    }
    catch (pika::exception const& e)
    {
        PIKA_TEST_EQ(e.get_error(), pika::broken_promise);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }
}

void test_assign_to_void()
{
    pika::future<void> f1 = pika::make_ready_future(42);
    f1.get();

    pika::shared_future<void> f2 = pika::make_ready_future(42).share();
    f2.get();
}

///////////////////////////////////////////////////////////////////////////////
using pika::program_options::options_description;
using pika::program_options::variables_map;

int pika_main(variables_map&)
{
    {
        test_store_value_from_thread();
        test_store_exception();
        test_initial_state();
        test_waiting_future();
        test_cannot_get_future_twice();
        test_set_value_updates_future_status();
        test_set_value_can_be_retrieved();
        test_set_value_can_be_moved();
        test_future_from_packaged_task_is_waiting();
        test_invoking_a_packaged_task_populates_future();
        test_invoking_a_packaged_task_twice_throws();
        test_cannot_get_future_twice_from_task();
        test_task_stores_exception_if_function_throws();
        test_void_promise();
        test_reference_promise();
        test_task_returning_void();
        test_task_returning_reference();
        test_future_for_move_only_udt();
        test_future_for_string();
        test_wait_callback();
        test_wait_callback_with_timed_wait();
        test_packaged_task_can_be_moved();
        test_destroying_a_promise_stores_broken_promise();
        test_destroying_a_packaged_task_stores_broken_task();
        test_assign_to_void();
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
