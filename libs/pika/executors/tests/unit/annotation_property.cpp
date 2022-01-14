//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>

#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
#include <pika/local/execution.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/latch.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/properties.hpp>
#include <pika/modules/testing.hpp>

#include <cstdlib>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
std::string annotation;

void test_post_f(int passed_through, pika::lcos::local::latch& l)
{
    PIKA_TEST_EQ(passed_through, 42);

    annotation =
        pika::threads::get_thread_description(pika::threads::get_self_id())
            .get_description();

    l.count_down(1);
}

void test_post()
{
    using executor = pika::execution::parallel_executor;

    std::string desc("test_post");
    {
        auto exec = pika::experimental::prefer(
            pika::execution::experimental::with_annotation, executor{}, desc);

        pika::lcos::local::latch l(2);
        pika::parallel::execution::post(exec, &test_post_f, 42, std::ref(l));
        l.arrive_and_wait();

        PIKA_TEST_EQ(annotation, desc);
    }

    {
        annotation.clear();
        auto exec =
            pika::execution::experimental::with_annotation(executor{}, desc);

        pika::lcos::local::latch l(2);
        pika::parallel::execution::post(exec, &test_post_f, 42, std::ref(l));
        l.arrive_and_wait();

        PIKA_TEST_EQ(annotation, desc);
    }
}

pika::thread::id test(int passed_through)
{
    PIKA_TEST_EQ(passed_through, 42);

    annotation =
        pika::threads::get_thread_description(pika::threads::get_self_id())
            .get_description();

    return pika::this_thread::get_id();
}

void test_sync()
{
    using executor = pika::execution::parallel_executor;

    std::string desc("test_sync");
    {
        auto exec = pika::experimental::prefer(
            pika::execution::experimental::with_annotation, executor{}, desc);

        PIKA_TEST(pika::parallel::execution::sync_execute(exec, &test, 42) ==
            pika::this_thread::get_id());
        PIKA_TEST_EQ(annotation, desc);
    }

    {
        annotation.clear();
        auto exec =
            pika::execution::experimental::with_annotation(executor{}, desc);

        PIKA_TEST(pika::parallel::execution::sync_execute(exec, &test, 42) ==
            pika::this_thread::get_id());
        PIKA_TEST_EQ(annotation, desc);
    }
}

void test_async()
{
    using executor = pika::execution::parallel_executor;

    std::string desc("test_async");
    {
        auto exec = pika::experimental::prefer(
            pika::execution::experimental::with_annotation, executor{}, desc);

        PIKA_TEST(
            pika::parallel::execution::async_execute(exec, &test, 42).get() !=
            pika::this_thread::get_id());
        PIKA_TEST_EQ(annotation, desc);
    }

    {
        annotation.clear();
        auto exec =
            pika::execution::experimental::with_annotation(executor{}, desc);

        PIKA_TEST(
            pika::parallel::execution::async_execute(exec, &test, 42).get() !=
            pika::this_thread::get_id());
        PIKA_TEST_EQ(annotation, desc);
    }
}

///////////////////////////////////////////////////////////////////////////////
pika::thread::id test_f(pika::future<void> f, int passed_through)
{
    PIKA_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    PIKA_TEST_EQ(passed_through, 42);

    annotation =
        pika::threads::get_thread_description(pika::threads::get_self_id())
            .get_description();

    return pika::this_thread::get_id();
}

void test_then()
{
    using executor = pika::execution::parallel_executor;

    pika::future<void> f = pika::make_ready_future();

    std::string desc("test_then");
    {
        auto exec = pika::experimental::prefer(
            pika::execution::experimental::with_annotation, executor{}, desc);

        PIKA_TEST(pika::parallel::execution::then_execute(exec, &test_f, f, 42)
                     .get() != pika::this_thread::get_id());
        PIKA_TEST_EQ(annotation, desc);
    }

    {
        annotation.clear();
        auto exec =
            pika::execution::experimental::with_annotation(executor{}, desc);

        PIKA_TEST(pika::parallel::execution::then_execute(exec, &test_f, f, 42)
                     .get() != pika::this_thread::get_id());
        PIKA_TEST_EQ(annotation, desc);
    }
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test(int seq, pika::thread::id tid, int passed_through)    //-V813
{
    PIKA_TEST_NEQ(tid, pika::this_thread::get_id());
    PIKA_TEST_EQ(passed_through, 42);

    if (seq == 0)
    {
        annotation =
            pika::threads::get_thread_description(pika::threads::get_self_id())
                .get_description();
    }
}

void test_bulk_sync()
{
    using executor = pika::execution::parallel_executor;

    pika::thread::id tid = pika::this_thread::get_id();

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    std::string desc("test_bulk_sync");
    {
        auto exec = pika::experimental::prefer(
            pika::execution::experimental::with_annotation, executor{}, desc);

        pika::parallel::execution::bulk_sync_execute(
            exec, pika::util::bind(&bulk_test, _1, tid, _2), 107, 42);
        PIKA_TEST_EQ(annotation, desc);

        annotation.clear();
        pika::parallel::execution::bulk_sync_execute(
            exec, &bulk_test, 107, tid, 42);
        PIKA_TEST_EQ(annotation, desc);
    }

    {
        auto exec =
            pika::execution::experimental::with_annotation(executor{}, desc);

        annotation.clear();
        pika::parallel::execution::bulk_sync_execute(
            exec, pika::util::bind(&bulk_test, _1, tid, _2), 107, 42);
        PIKA_TEST_EQ(annotation, desc);

        annotation.clear();
        pika::parallel::execution::bulk_sync_execute(
            exec, &bulk_test, 107, tid, 42);
        PIKA_TEST_EQ(annotation, desc);
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_bulk_async()
{
    using executor = pika::execution::parallel_executor;

    pika::thread::id tid = pika::this_thread::get_id();

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    std::string desc("test_bulk_async");
    {
        auto exec = pika::experimental::prefer(
            pika::execution::experimental::with_annotation, executor{}, desc);

        pika::when_all(pika::parallel::execution::bulk_async_execute(exec,
                          pika::util::bind(&bulk_test, _1, tid, _2), 107, 42))
            .get();
        PIKA_TEST_EQ(annotation, desc);

        annotation.clear();
        pika::when_all(pika::parallel::execution::bulk_async_execute(
                          exec, &bulk_test, 107, tid, 42))
            .get();
        PIKA_TEST_EQ(annotation, desc);
    }

    {
        auto exec =
            pika::execution::experimental::with_annotation(executor{}, desc);

        annotation.clear();
        pika::when_all(pika::parallel::execution::bulk_async_execute(exec,
                          pika::util::bind(&bulk_test, _1, tid, _2), 107, 42))
            .get();
        PIKA_TEST_EQ(annotation, desc);

        annotation.clear();
        pika::when_all(pika::parallel::execution::bulk_async_execute(
                          exec, &bulk_test, 107, tid, 42))
            .get();
        PIKA_TEST_EQ(annotation, desc);
    }
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test_f(int seq, pika::shared_future<void> f, pika::thread::id tid,
    int passed_through)    //-V813
{
    PIKA_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    PIKA_TEST_NEQ(tid, pika::this_thread::get_id());
    PIKA_TEST_EQ(passed_through, 42);

    if (seq == 0)
    {
        annotation =
            pika::threads::get_thread_description(pika::threads::get_self_id())
                .get_description();
    }
}

void test_bulk_then()
{
    using executor = pika::execution::parallel_executor;

    pika::thread::id tid = pika::this_thread::get_id();

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;
    using pika::util::placeholders::_3;

    pika::shared_future<void> f = pika::make_ready_future();

    std::string desc("test_bulk_then");
    {
        auto exec = pika::experimental::prefer(
            pika::execution::experimental::with_annotation, executor{}, desc);

        pika::parallel::execution::bulk_then_execute(
            exec, pika::util::bind(&bulk_test_f, _1, _2, tid, _3), 107, f, 42)
            .get();
        PIKA_TEST_EQ(annotation, desc);

        annotation.clear();
        pika::parallel::execution::bulk_then_execute(
            exec, &bulk_test_f, 107, f, tid, 42)
            .get();
        PIKA_TEST_EQ(annotation, desc);
    }

    {
        auto exec =
            pika::execution::experimental::with_annotation(executor{}, desc);

        annotation.clear();
        pika::parallel::execution::bulk_then_execute(
            exec, pika::util::bind(&bulk_test_f, _1, _2, tid, _3), 107, f, 42)
            .get();
        PIKA_TEST_EQ(annotation, desc);

        annotation.clear();
        pika::parallel::execution::bulk_then_execute(
            exec, &bulk_test_f, 107, f, tid, 42)
            .get();
        PIKA_TEST_EQ(annotation, desc);
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_post_policy_prefer()
{
    std::string desc("test_post_policy_prefer");
    auto policy =
        pika::experimental::prefer(pika::execution::experimental::with_annotation,
            pika::execution::par, desc);

    pika::lcos::local::latch l(2);
    pika::parallel::execution::post(
        policy.executor(), &test_post_f, 42, std::ref(l));
    l.arrive_and_wait();

    PIKA_TEST_EQ(annotation, desc);
}

void test_post_policy()
{
    std::string desc("test_post_policy");
    auto policy = pika::execution::experimental::with_annotation(
        pika::execution::par, desc);

    pika::lcos::local::latch l(2);
    pika::parallel::execution::post(
        policy.executor(), &test_post_f, 42, std::ref(l));
    l.arrive_and_wait();

    PIKA_TEST_EQ(annotation, desc);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    test_post();

    test_sync();
    test_async();
    test_then();

    test_bulk_sync();
    test_bulk_async();
    test_bulk_then();

    test_post_policy_prefer();
    test_post_policy();

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
#else
int main()
{
    return 0;
}
#endif
