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
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
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
    pika::shared_future<int> fi2(pi2.get_future());
    pika::thread t(&set_promise_thread, &pi2);
    int j = fi2.get();
    PIKA_TEST_EQ(j, 42);
    PIKA_TEST(fi2.is_ready());
    PIKA_TEST(fi2.has_value());
    PIKA_TEST(!fi2.has_exception());
    t.join();
}

///////////////////////////////////////////////////////////////////////////////
void test_store_exception()
{
    pika::lcos::local::promise<int> pi3;
    pika::shared_future<int> fi3 = pi3.get_future();
    pika::thread t(&set_promise_exception_thread, &pi3);
    try
    {
        fi3.get();
        PIKA_TEST(false);
    }
    catch (my_exception)
    {
        PIKA_TEST(true);
    }

    PIKA_TEST(fi3.is_ready());
    PIKA_TEST(!fi3.has_value());
    PIKA_TEST(fi3.has_exception());
    t.join();
}

///////////////////////////////////////////////////////////////////////////////
void test_initial_state()
{
    pika::shared_future<int> fi;
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
    pika::shared_future<int> fi;
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
    pika::shared_future<int> fi;
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
    pika::shared_future<int> fi;
    fi = pi.get_future();

    pi.set_value(42);

    int i = fi.get();
    PIKA_TEST_EQ(i, 42);
    PIKA_TEST(fi.is_ready());
    PIKA_TEST(fi.has_value());
    PIKA_TEST(!fi.has_exception());
}

///////////////////////////////////////////////////////////////////////////////
void test_set_value_can_be_moved()
{
    pika::lcos::local::promise<int> pi;
    pika::shared_future<int> fi;
    fi = pi.get_future();

    pi.set_value(42);

    int i = 0;
    PIKA_TEST(i = fi.get());
    PIKA_TEST_EQ(i, 42);
    PIKA_TEST(fi.is_ready());
    PIKA_TEST(fi.has_value());
    PIKA_TEST(!fi.has_exception());
}

///////////////////////////////////////////////////////////////////////////////
void test_future_from_packaged_task_is_waiting()
{
    pika::lcos::local::packaged_task<int()> pt(make_int);
    pika::shared_future<int> fi = pt.get_future();

    PIKA_TEST(!fi.is_ready());
    PIKA_TEST(!fi.has_value());
    PIKA_TEST(!fi.has_exception());
}

///////////////////////////////////////////////////////////////////////////////
void test_invoking_a_packaged_task_populates_future()
{
    pika::lcos::local::packaged_task<int()> pt(make_int);
    pika::shared_future<int> fi = pt.get_future();

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
    pika::shared_future<int> fi = pt.get_future();

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
    pika::shared_future<void> f = p.get_future();

    p.set_value();
    PIKA_TEST(f.is_ready());
    PIKA_TEST(f.has_value());
    PIKA_TEST(!f.has_exception());
}

void test_reference_promise()
{
    pika::lcos::local::promise<int&> p;
    pika::shared_future<int&> f = p.get_future();
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
    pika::shared_future<void> fi = pt.get_future();

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
    pika::shared_future<int&> fi = pt.get_future();

    pt();

    PIKA_TEST(fi.is_ready());
    PIKA_TEST(fi.has_value());
    PIKA_TEST(!fi.has_exception());
    int& i = fi.get();
    PIKA_TEST_EQ(&i, &global_ref_target);
}

void test_shared_future()
{
    pika::lcos::local::packaged_task<int()> pt(make_int);
    pika::shared_future<int> fi = pt.get_future();

    pika::shared_future<int> sf(std::move(fi));

    pt();

    PIKA_TEST(sf.is_ready());
    PIKA_TEST(sf.has_value());
    PIKA_TEST(!sf.has_exception());

    int i = sf.get();
    PIKA_TEST_EQ(i, 42);
}

void test_copies_of_shared_future_become_ready_together()
{
    pika::lcos::local::packaged_task<int()> pt(make_int);
    pika::shared_future<int> fi = pt.get_future();

    pika::shared_future<int> sf1(std::move(fi));
    pika::shared_future<int> sf2(sf1);
    pika::shared_future<int> sf3;

    sf3 = sf1;
    PIKA_TEST(!sf1.is_ready());
    PIKA_TEST(!sf2.is_ready());
    PIKA_TEST(!sf3.is_ready());

    pt();

    PIKA_TEST(sf1.is_ready());
    PIKA_TEST(sf1.has_value());
    PIKA_TEST(!sf1.has_exception());
    int i = sf1.get();
    PIKA_TEST_EQ(i, 42);

    PIKA_TEST(sf2.is_ready());
    PIKA_TEST(sf2.has_value());
    PIKA_TEST(!sf2.has_exception());
    i = sf2.get();
    PIKA_TEST_EQ(i, 42);

    PIKA_TEST(sf3.is_ready());
    PIKA_TEST(sf3.has_value());
    PIKA_TEST(!sf3.has_exception());
    i = sf3.get();
    PIKA_TEST_EQ(i, 42);
}

void test_shared_future_can_be_move_assigned_from_shared_future()
{
    pika::lcos::local::packaged_task<int()> pt(make_int);
    pika::shared_future<int> fi = pt.get_future();

    pika::shared_future<int> sf;
    sf = std::move(fi);
    PIKA_TEST(!fi.valid());    // NOLINT

    PIKA_TEST(!sf.is_ready());
    PIKA_TEST(!sf.has_value());
    PIKA_TEST(!sf.has_exception());
}

void test_shared_future_void()
{
    pika::lcos::local::packaged_task<void()> pt(do_nothing);
    pika::shared_future<void> fi = pt.get_future();

    pika::shared_future<void> sf(std::move(fi));
    PIKA_TEST(!fi.valid());    // NOLINT

    pt();

    PIKA_TEST(sf.is_ready());
    PIKA_TEST(sf.has_value());
    PIKA_TEST(!sf.has_exception());
    sf.get();
}

void test_shared_future_ref()
{
    pika::lcos::local::promise<int&> p;
    pika::shared_future<int&> f(p.get_future());
    int i = 42;
    p.set_value(i);
    PIKA_TEST(f.is_ready());
    PIKA_TEST(f.has_value());
    PIKA_TEST(!f.has_exception());
    PIKA_TEST_EQ(&f.get(), &i);
}

void test_shared_future_for_string()
{
    pika::lcos::local::promise<std::string> pt;
    pika::shared_future<std::string> fi1 = pt.get_future();

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

void wait_callback(pika::shared_future<int>)
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
        PIKA_TEST(false);
    }
}

void test_wait_callback()
{
    callback_called = 0;
    pika::lcos::local::promise<int> pi;
    pika::shared_future<int> fi = pi.get_future();

    pika::future<void> cbf = fi.then(&wait_callback);
    pika::thread t(&promise_set_value, std::ref(pi));

    fi.wait();
    cbf.wait();

    t.join();

    PIKA_TEST_EQ(callback_called, 1U);
    PIKA_TEST_EQ(fi.get(), 42);
    fi.wait();
    fi.wait();
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
    pika::shared_future<int> fi = pi.get_future();

    pika::shared_future<void> fv =
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
    pika::shared_future<int> fi = pt.get_future();
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
    pika::shared_future<int> f;

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
    pika::shared_future<int> f;

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

///////////////////////////////////////////////////////////////////////////////
int make_int_slowly()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return 42;
}

void test_wait_for_either_of_two_futures_1()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::shared_future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2(pt2.get_future());

    pt1();

    pika::future<pika::when_any_result<
        pika::tuple<pika::shared_future<int>, pika::shared_future<int>>>>
        r = pika::when_any(f1, f2);
    pika::tuple<pika::shared_future<int>, pika::shared_future<int>> t =
        r.get().futures;

    PIKA_TEST(f1.is_ready());
    PIKA_TEST(!f2.is_ready());
    PIKA_TEST_EQ(f1.get(), 42);

    PIKA_TEST(pika::get<0>(t).is_ready());
    PIKA_TEST_EQ(pika::get<0>(t).get(), 42);
}

void test_wait_for_either_of_two_futures_2()
{
    pika::lcos::local::packaged_task<int()> pt(make_int_slowly);
    pika::shared_future<int> f1(pt.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2(pt2.get_future());

    pt2();

    pika::future<pika::when_any_result<
        pika::tuple<pika::shared_future<int>, pika::shared_future<int>>>>
        r = pika::when_any(f1, f2);
    pika::tuple<pika::shared_future<int>, pika::shared_future<int>> t =
        r.get().futures;

    PIKA_TEST(!f1.is_ready());
    PIKA_TEST(f2.is_ready());
    PIKA_TEST_EQ(f2.get(), 42);

    PIKA_TEST(pika::get<1>(t).is_ready());
    PIKA_TEST_EQ(pika::get<1>(t).get(), 42);
}

void test_wait_for_either_of_two_futures_list_1()
{
    std::vector<pika::shared_future<int>> futures;
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    futures.push_back(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    futures.push_back(pt2.get_future());

    pt1();

    pika::future<pika::when_any_result<std::vector<pika::shared_future<int>>>> r =
        pika::when_any(futures);
    pika::when_any_result<std::vector<pika::shared_future<int>>> raw = r.get();

    PIKA_TEST_EQ(raw.index, 0u);

    std::vector<pika::shared_future<int>> t = std::move(raw.futures);

    PIKA_TEST(futures[0].is_ready());
    PIKA_TEST(!futures[1].is_ready());
    PIKA_TEST_EQ(futures[0].get(), 42);

    PIKA_TEST(t[0].is_ready());
    PIKA_TEST_EQ(t[0].get(), 42);
}

void test_wait_for_either_of_two_futures_list_2()
{
    std::vector<pika::shared_future<int>> futures;
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    futures.push_back(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    futures.push_back(pt2.get_future());

    pt2();

    pika::future<pika::when_any_result<std::vector<pika::shared_future<int>>>> r =
        pika::when_any(futures);
    pika::when_any_result<std::vector<pika::shared_future<int>>> raw = r.get();

    PIKA_TEST_EQ(raw.index, 1u);

    std::vector<pika::shared_future<int>> t = std::move(raw.futures);

    PIKA_TEST(!futures[0].is_ready());
    PIKA_TEST(futures[1].is_ready());
    PIKA_TEST_EQ(futures[1].get(), 42);

    PIKA_TEST(t[1].is_ready());
    PIKA_TEST_EQ(t[1].get(), 42);
}

void test_wait_for_either_of_three_futures_1()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::shared_future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::shared_future<int> f3(pt3.get_future());

    pt1();

    pika::future<pika::when_any_result<pika::tuple<pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>>>>
        r = pika::when_any(f1, f2, f3);
    pika::tuple<pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>>
        t = r.get().futures;

    PIKA_TEST(f1.is_ready());
    PIKA_TEST(!f2.is_ready());
    PIKA_TEST(!f3.is_ready());
    PIKA_TEST_EQ(f1.get(), 42);

    PIKA_TEST(pika::get<0>(t).is_ready());
    PIKA_TEST_EQ(pika::get<0>(t).get(), 42);
}

void test_wait_for_either_of_three_futures_2()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::shared_future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::shared_future<int> f3(pt3.get_future());

    pt2();

    pika::future<pika::when_any_result<pika::tuple<pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>>>>
        r = pika::when_any(f1, f2, f3);
    pika::tuple<pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>>
        t = r.get().futures;

    PIKA_TEST(!f1.is_ready());
    PIKA_TEST(f2.is_ready());
    PIKA_TEST(!f3.is_ready());
    PIKA_TEST_EQ(f2.get(), 42);

    PIKA_TEST(pika::get<1>(t).is_ready());
    PIKA_TEST_EQ(pika::get<1>(t).get(), 42);
}

void test_wait_for_either_of_three_futures_3()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::shared_future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::shared_future<int> f3(pt3.get_future());

    pt3();

    pika::future<pika::when_any_result<pika::tuple<pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>>>>
        r = pika::when_any(f1, f2, f3);
    pika::tuple<pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>>
        t = r.get().futures;

    PIKA_TEST(!f1.is_ready());
    PIKA_TEST(!f2.is_ready());
    PIKA_TEST(f3.is_ready());
    PIKA_TEST_EQ(f3.get(), 42);

    PIKA_TEST(pika::get<2>(t).is_ready());
    PIKA_TEST_EQ(pika::get<2>(t).get(), 42);
}

void test_wait_for_either_of_four_futures_1()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::shared_future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::shared_future<int> f3(pt3.get_future());
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::shared_future<int> f4(pt4.get_future());

    pt1();

    pika::future<pika::when_any_result<
        pika::tuple<pika::shared_future<int>, pika::shared_future<int>,
            pika::shared_future<int>, pika::shared_future<int>>>>
        r = pika::when_any(f1, f2, f3, f4);
    pika::tuple<pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>>
        t = r.get().futures;

    PIKA_TEST(f1.is_ready());
    PIKA_TEST(!f2.is_ready());
    PIKA_TEST(!f3.is_ready());
    PIKA_TEST(!f4.is_ready());
    PIKA_TEST_EQ(f1.get(), 42);

    PIKA_TEST(pika::get<0>(t).is_ready());
    PIKA_TEST_EQ(pika::get<0>(t).get(), 42);
}

void test_wait_for_either_of_four_futures_2()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::shared_future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::shared_future<int> f3(pt3.get_future());
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::shared_future<int> f4(pt4.get_future());

    pt2();

    pika::future<pika::when_any_result<
        pika::tuple<pika::shared_future<int>, pika::shared_future<int>,
            pika::shared_future<int>, pika::shared_future<int>>>>
        r = pika::when_any(f1, f2, f3, f4);
    pika::tuple<pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>>
        t = r.get().futures;

    PIKA_TEST(!f1.is_ready());
    PIKA_TEST(f2.is_ready());
    PIKA_TEST(!f3.is_ready());
    PIKA_TEST(!f4.is_ready());
    PIKA_TEST_EQ(f2.get(), 42);

    PIKA_TEST(pika::get<1>(t).is_ready());
    PIKA_TEST_EQ(pika::get<1>(t).get(), 42);
}

void test_wait_for_either_of_four_futures_3()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::shared_future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::shared_future<int> f3(pt3.get_future());
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::shared_future<int> f4(pt4.get_future());

    pt3();

    pika::future<pika::when_any_result<
        pika::tuple<pika::shared_future<int>, pika::shared_future<int>,
            pika::shared_future<int>, pika::shared_future<int>>>>
        r = pika::when_any(f1, f2, f3, f4);
    pika::tuple<pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>>
        t = r.get().futures;

    PIKA_TEST(!f1.is_ready());
    PIKA_TEST(!f2.is_ready());
    PIKA_TEST(f3.is_ready());
    PIKA_TEST(!f4.is_ready());
    PIKA_TEST_EQ(f3.get(), 42);

    PIKA_TEST(pika::get<2>(t).is_ready());
    PIKA_TEST_EQ(pika::get<2>(t).get(), 42);
}

void test_wait_for_either_of_four_futures_4()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::shared_future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::shared_future<int> f3(pt3.get_future());
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::shared_future<int> f4(pt4.get_future());

    pt4();

    pika::future<pika::when_any_result<
        pika::tuple<pika::shared_future<int>, pika::shared_future<int>,
            pika::shared_future<int>, pika::shared_future<int>>>>
        r = pika::when_any(f1, f2, f3, f4);
    pika::tuple<pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>>
        t = r.get().futures;

    PIKA_TEST(!f1.is_ready());
    PIKA_TEST(!f2.is_ready());
    PIKA_TEST(!f3.is_ready());
    PIKA_TEST(f4.is_ready());
    PIKA_TEST_EQ(f4.get(), 42);

    PIKA_TEST(pika::get<3>(t).is_ready());
    PIKA_TEST_EQ(pika::get<3>(t).get(), 42);
}

void test_wait_for_either_of_five_futures_1_from_list()
{
    std::vector<pika::shared_future<int>> futures;

    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::shared_future<int> f1(pt1.get_future());
    futures.push_back(f1);
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2(pt2.get_future());
    futures.push_back(f2);
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::shared_future<int> f3(pt3.get_future());
    futures.push_back(f3);
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::shared_future<int> f4(pt4.get_future());
    futures.push_back(f4);
    pika::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    pika::shared_future<int> f5(pt5.get_future());
    futures.push_back(f5);

    pt1();

    pika::future<pika::when_any_result<std::vector<pika::shared_future<int>>>> r =
        pika::when_any(futures);
    pika::when_any_result<std::vector<pika::shared_future<int>>> raw = r.get();

    PIKA_TEST_EQ(raw.index, 0u);

    std::vector<pika::shared_future<int>> t = std::move(raw.futures);

    PIKA_TEST(f1.is_ready());
    PIKA_TEST(!f2.is_ready());
    PIKA_TEST(!f3.is_ready());
    PIKA_TEST(!f4.is_ready());
    PIKA_TEST(!f5.is_ready());
    PIKA_TEST_EQ(f1.get(), 42);

    PIKA_TEST(t[0].is_ready());
    PIKA_TEST_EQ(t[0].get(), 42);
}

void test_wait_for_either_of_five_futures_1_from_list_iterators()
{
    std::vector<pika::shared_future<int>> futures;

    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::shared_future<int> f1(pt1.get_future());
    futures.push_back(f1);
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2(pt2.get_future());
    futures.push_back(f2);
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::shared_future<int> f3(pt3.get_future());
    futures.push_back(f3);
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::shared_future<int> f4(pt4.get_future());
    futures.push_back(f4);
    pika::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    pika::shared_future<int> f5(pt5.get_future());
    futures.push_back(f5);

    pt1();

    pika::future<pika::when_any_result<std::vector<pika::shared_future<int>>>> r =
        pika::when_any(futures.begin(), futures.end());
    pika::when_any_result<std::vector<pika::shared_future<int>>> raw = r.get();

    PIKA_TEST_EQ(raw.index, 0u);

    std::vector<pika::shared_future<int>> t = std::move(raw.futures);

    PIKA_TEST(f1.is_ready());
    PIKA_TEST(!f2.is_ready());
    PIKA_TEST(!f3.is_ready());
    PIKA_TEST(!f4.is_ready());
    PIKA_TEST(!f5.is_ready());
    PIKA_TEST_EQ(f1.get(), 42);

    PIKA_TEST(t[0].is_ready());
    PIKA_TEST_EQ(t[0].get(), 42);
}

void test_wait_for_either_of_five_futures_1()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::shared_future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::shared_future<int> f3(pt3.get_future());
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::shared_future<int> f4(pt4.get_future());
    pika::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    pika::shared_future<int> f5(pt5.get_future());

    pt1();

    pika::future<pika::when_any_result<pika::tuple<pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>>>>
        r = pika::when_any(f1, f2, f3, f4, f5);
    pika::tuple<pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>>
        t = r.get().futures;

    PIKA_TEST(f1.is_ready());
    PIKA_TEST(!f2.is_ready());
    PIKA_TEST(!f3.is_ready());
    PIKA_TEST(!f4.is_ready());
    PIKA_TEST(!f5.is_ready());
    PIKA_TEST_EQ(f1.get(), 42);

    PIKA_TEST(pika::get<0>(t).is_ready());
    PIKA_TEST_EQ(pika::get<0>(t).get(), 42);
}

void test_wait_for_either_of_five_futures_2()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::shared_future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::shared_future<int> f3(pt3.get_future());
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::shared_future<int> f4(pt4.get_future());
    pika::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    pika::shared_future<int> f5(pt5.get_future());

    pt2();

    pika::future<pika::when_any_result<pika::tuple<pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>>>>
        r = pika::when_any(f1, f2, f3, f4, f5);
    pika::tuple<pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>>
        t = r.get().futures;

    PIKA_TEST(!f1.is_ready());
    PIKA_TEST(f2.is_ready());
    PIKA_TEST(!f3.is_ready());
    PIKA_TEST(!f4.is_ready());
    PIKA_TEST(!f5.is_ready());
    PIKA_TEST_EQ(f2.get(), 42);

    PIKA_TEST(pika::get<1>(t).is_ready());
    PIKA_TEST_EQ(pika::get<1>(t).get(), 42);
}

void test_wait_for_either_of_five_futures_3()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::shared_future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::shared_future<int> f3(pt3.get_future());
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::shared_future<int> f4(pt4.get_future());
    pika::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    pika::shared_future<int> f5(pt5.get_future());

    pt3();

    pika::future<pika::when_any_result<pika::tuple<pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>>>>
        r = pika::when_any(f1, f2, f3, f4, f5);
    pika::tuple<pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>>
        t = r.get().futures;

    PIKA_TEST(!f1.is_ready());
    PIKA_TEST(!f2.is_ready());
    PIKA_TEST(f3.is_ready());
    PIKA_TEST(!f4.is_ready());
    PIKA_TEST(!f5.is_ready());
    PIKA_TEST_EQ(f3.get(), 42);

    PIKA_TEST(pika::get<2>(t).is_ready());
    PIKA_TEST_EQ(pika::get<2>(t).get(), 42);
}

void test_wait_for_either_of_five_futures_4()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::shared_future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::shared_future<int> f3(pt3.get_future());
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::shared_future<int> f4(pt4.get_future());
    pika::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    pika::shared_future<int> f5(pt5.get_future());

    pt4();

    pika::future<pika::when_any_result<pika::tuple<pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>>>>
        r = pika::when_any(f1, f2, f3, f4, f5);
    pika::tuple<pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>>
        t = r.get().futures;

    PIKA_TEST(!f1.is_ready());
    PIKA_TEST(!f2.is_ready());
    PIKA_TEST(!f3.is_ready());
    PIKA_TEST(f4.is_ready());
    PIKA_TEST(!f5.is_ready());
    PIKA_TEST_EQ(f4.get(), 42);

    PIKA_TEST(pika::get<3>(t).is_ready());
    PIKA_TEST_EQ(pika::get<3>(t).get(), 42);
}

void test_wait_for_either_of_five_futures_5()
{
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::shared_future<int> f1(pt1.get_future());
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2(pt2.get_future());
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::shared_future<int> f3(pt3.get_future());
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::shared_future<int> f4(pt4.get_future());
    pika::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    pika::shared_future<int> f5(pt5.get_future());

    pt5();

    pika::future<pika::when_any_result<pika::tuple<pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>>>>
        r = pika::when_any(f1, f2, f3, f4, f5);
    pika::tuple<pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>>
        t = r.get().futures;

    PIKA_TEST(!f1.is_ready());
    PIKA_TEST(!f2.is_ready());
    PIKA_TEST(!f3.is_ready());
    PIKA_TEST(!f4.is_ready());
    PIKA_TEST(f5.is_ready());
    PIKA_TEST_EQ(f5.get(), 42);

    PIKA_TEST(pika::get<4>(t).is_ready());
    PIKA_TEST_EQ(pika::get<4>(t).get(), 42);
}

///////////////////////////////////////////////////////////////////////////////
// void test_wait_for_either_invokes_callbacks()
// {
//     callback_called = 0;
//     pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
//     pika::shared_future<int> fi = pt1.get_future();
//     pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
//     pika::shared_future<int> fi2 = pt2.get_future();
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
//         pika::shared_future<int> futures[count];
//         for(unsigned j = 0; j < count; ++j)
//         {
//             tasks[j] =
//               std::move(pika::lcos::local::packaged_task<int()>(make_int_slowly));
//             futures[j] = tasks[j].get_future();
//         }
//         pika::thread t(std::move(tasks[i]));
//
//         pika::lcos::wait_any(futures, futures);
//
//         pika::shared_future<int>* const future =
//              boost::wait_for_any(futures, futures+count);
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

void test_wait_for_all_from_list()
{
    unsigned const count = 10;
    std::vector<pika::shared_future<int>> futures;
    for (unsigned j = 0; j < count; ++j)
    {
        pika::lcos::local::futures_factory<int()> task(make_int_slowly);
        futures.push_back(task.get_future());
        task.apply();
    }

    pika::future<std::vector<pika::shared_future<int>>> r =
        pika::when_all(futures);

    std::vector<pika::shared_future<int>> result = r.get();

    PIKA_TEST_EQ(futures.size(), result.size());
    for (unsigned j = 0; j < count; ++j)
    {
        PIKA_TEST(futures[j].is_ready());
        PIKA_TEST(result[j].is_ready());
    }
}

void test_wait_for_all_from_list_iterators()
{
    unsigned const count = 10;
    std::vector<pika::shared_future<int>> futures;
    for (unsigned j = 0; j < count; ++j)
    {
        pika::lcos::local::futures_factory<int()> task(make_int_slowly);
        futures.push_back(task.get_future());
        task.apply();
    }

    pika::future<std::vector<pika::shared_future<int>>> r =
        pika::when_all(futures.begin(), futures.end());

    std::vector<pika::shared_future<int>> result = r.get();

    PIKA_TEST_EQ(futures.size(), result.size());
    for (unsigned j = 0; j < count; ++j)
    {
        PIKA_TEST(futures[j].is_ready());
        PIKA_TEST(result[j].is_ready());
    }
}

void test_wait_for_all_two_futures()
{
    pika::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    pika::shared_future<int> f1 = pt1.get_future();
    pt1.apply();
    pika::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2 = pt2.get_future();
    pt2.apply();

    typedef pika::tuple<pika::shared_future<int>, pika::shared_future<int>>
        result_type;
    pika::future<result_type> r = pika::when_all(f1, f2);

    result_type result = r.get();

    PIKA_TEST(pika::get<0>(result).is_ready());
    PIKA_TEST(pika::get<1>(result).is_ready());
    PIKA_TEST(f1.is_ready());
    PIKA_TEST(f2.is_ready());
}

void test_wait_for_all_three_futures()
{
    pika::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    pika::shared_future<int> f1 = pt1.get_future();
    pt1.apply();
    pika::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2 = pt2.get_future();
    pt2.apply();
    pika::lcos::local::futures_factory<int()> pt3(make_int_slowly);
    pika::shared_future<int> f3 = pt3.get_future();
    pt3.apply();

    typedef pika::tuple<pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>>
        result_type;
    pika::future<result_type> r = pika::when_all(f1, f2, f3);

    result_type result = r.get();

    PIKA_TEST(pika::get<0>(result).is_ready());
    PIKA_TEST(pika::get<1>(result).is_ready());
    PIKA_TEST(pika::get<2>(result).is_ready());
    PIKA_TEST(f1.is_ready());
    PIKA_TEST(f2.is_ready());
    PIKA_TEST(f3.is_ready());
}

void test_wait_for_all_four_futures()
{
    pika::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    pika::shared_future<int> f1 = pt1.get_future();
    pt1.apply();
    pika::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2 = pt2.get_future();
    pt2.apply();
    pika::lcos::local::futures_factory<int()> pt3(make_int_slowly);
    pika::shared_future<int> f3 = pt3.get_future();
    pt3.apply();
    pika::lcos::local::futures_factory<int()> pt4(make_int_slowly);
    pika::shared_future<int> f4 = pt4.get_future();
    pt4.apply();

    typedef pika::tuple<pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>>
        result_type;
    pika::future<result_type> r = pika::when_all(f1, f2, f3, f4);

    result_type result = r.get();

    PIKA_TEST(pika::get<0>(result).is_ready());
    PIKA_TEST(pika::get<1>(result).is_ready());
    PIKA_TEST(pika::get<2>(result).is_ready());
    PIKA_TEST(pika::get<3>(result).is_ready());
    PIKA_TEST(f1.is_ready());
    PIKA_TEST(f2.is_ready());
    PIKA_TEST(f3.is_ready());
    PIKA_TEST(f4.is_ready());
}

void test_wait_for_all_five_futures()
{
    pika::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    pika::shared_future<int> f1 = pt1.get_future();
    pt1.apply();
    pika::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2 = pt2.get_future();
    pt2.apply();
    pika::lcos::local::futures_factory<int()> pt3(make_int_slowly);
    pika::shared_future<int> f3 = pt3.get_future();
    pt3.apply();
    pika::lcos::local::futures_factory<int()> pt4(make_int_slowly);
    pika::shared_future<int> f4 = pt4.get_future();
    pt4.apply();
    pika::lcos::local::futures_factory<int()> pt5(make_int_slowly);
    pika::shared_future<int> f5 = pt5.get_future();
    pt5.apply();

    typedef pika::tuple<pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>>
        result_type;
    pika::future<result_type> r = pika::when_all(f1, f2, f3, f4, f5);

    result_type result = r.get();

    PIKA_TEST(pika::get<0>(result).is_ready());
    PIKA_TEST(pika::get<1>(result).is_ready());
    PIKA_TEST(pika::get<2>(result).is_ready());
    PIKA_TEST(pika::get<3>(result).is_ready());
    PIKA_TEST(pika::get<4>(result).is_ready());
    PIKA_TEST(f1.is_ready());
    PIKA_TEST(f2.is_ready());
    PIKA_TEST(f3.is_ready());
    PIKA_TEST(f4.is_ready());
    PIKA_TEST(f5.is_ready());
}

void test_wait_for_two_out_of_five_futures()
{
    unsigned const count = 2;

    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::shared_future<int> f1 = pt1.get_future();
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2 = pt2.get_future();
    pt2();
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::shared_future<int> f3 = pt3.get_future();
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::shared_future<int> f4 = pt4.get_future();
    pt4();
    pika::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    pika::shared_future<int> f5 = pt5.get_future();

    typedef pika::when_some_result<pika::tuple<pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>>>
        result_type;
    pika::future<result_type> r = pika::when_some(count, f1, f2, f3, f4, f5);

    result_type result = r.get();

    PIKA_TEST(!f1.is_ready());
    PIKA_TEST(f2.is_ready());
    PIKA_TEST(!f3.is_ready());
    PIKA_TEST(f4.is_ready());
    PIKA_TEST(!f5.is_ready());

    PIKA_TEST_EQ(result.indices.size(), count);
    PIKA_TEST(!pika::get<0>(result.futures).is_ready());
    PIKA_TEST(pika::get<1>(result.futures).is_ready());
    PIKA_TEST(!pika::get<2>(result.futures).is_ready());
    PIKA_TEST(pika::get<3>(result.futures).is_ready());
    PIKA_TEST(!pika::get<4>(result.futures).is_ready());
}

void test_wait_for_three_out_of_five_futures()
{
    unsigned const count = 3;

    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    pika::shared_future<int> f1 = pt1.get_future();
    pt1();
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    pika::shared_future<int> f2 = pt2.get_future();
    pika::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    pika::shared_future<int> f3 = pt3.get_future();
    pt3();
    pika::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    pika::shared_future<int> f4 = pt4.get_future();
    pika::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    pika::shared_future<int> f5 = pt5.get_future();
    pt5();

    typedef pika::when_some_result<pika::tuple<pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>,
        pika::shared_future<int>, pika::shared_future<int>>>
        result_type;
    pika::future<result_type> r = pika::when_some(count, f1, f2, f3, f4, f5);

    result_type result = r.get();

    PIKA_TEST(f1.is_ready());
    PIKA_TEST(!f2.is_ready());
    PIKA_TEST(f3.is_ready());
    PIKA_TEST(!f4.is_ready());
    PIKA_TEST(f5.is_ready());

    PIKA_TEST_EQ(result.indices.size(), count);
    PIKA_TEST(pika::get<0>(result.futures).is_ready());
    PIKA_TEST(!pika::get<1>(result.futures).is_ready());
    PIKA_TEST(pika::get<2>(result.futures).is_ready());
    PIKA_TEST(!pika::get<3>(result.futures).is_ready());
    PIKA_TEST(pika::get<4>(result.futures).is_ready());
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
        test_shared_future();
        test_copies_of_shared_future_become_ready_together();
        test_shared_future_can_be_move_assigned_from_shared_future();
        test_shared_future_void();
        test_shared_future_ref();
        test_shared_future_for_string();
        test_wait_callback();
        test_wait_callback_with_timed_wait();
        test_packaged_task_can_be_moved();
        test_destroying_a_promise_stores_broken_promise();
        test_destroying_a_packaged_task_stores_broken_task();
        test_wait_for_either_of_two_futures_1();
        test_wait_for_either_of_two_futures_2();
        test_wait_for_either_of_two_futures_list_1();
        test_wait_for_either_of_two_futures_list_2();
        test_wait_for_either_of_three_futures_1();
        test_wait_for_either_of_three_futures_2();
        test_wait_for_either_of_three_futures_3();
        test_wait_for_either_of_four_futures_1();
        test_wait_for_either_of_four_futures_2();
        test_wait_for_either_of_four_futures_3();
        test_wait_for_either_of_four_futures_4();
        test_wait_for_either_of_five_futures_1_from_list();
        test_wait_for_either_of_five_futures_1_from_list_iterators();
        test_wait_for_either_of_five_futures_1();
        test_wait_for_either_of_five_futures_2();
        test_wait_for_either_of_five_futures_3();
        test_wait_for_either_of_five_futures_4();
        test_wait_for_either_of_five_futures_5();
        //         test_wait_for_either_invokes_callbacks();
        //         test_wait_for_any_from_range();
        test_wait_for_all_from_list();
        test_wait_for_all_from_list_iterators();
        test_wait_for_all_two_futures();
        test_wait_for_all_three_futures();
        test_wait_for_all_four_futures();
        test_wait_for_all_five_futures();
        test_wait_for_two_out_of_five_futures();
        test_wait_for_three_out_of_five_futures();
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
    std::vector<std::string> const cfg = {"pika.os_threads=4"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.desc_cmdline = cmdline;
    init_args.cfg = cfg;

    return pika::local::init(pika_main, argc, argv, init_args);
}
