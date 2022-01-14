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
#include <string>
#include <type_traits>
#include <utility>
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

template <typename Executor>
void test_post(Executor&& executor)
{
    std::string desc("test_post");
    {
        auto exec = pika::experimental::prefer(
            pika::execution::experimental::with_annotation, executor, desc);

        pika::lcos::local::latch l(2);
        pika::parallel::execution::post(exec, &test_post_f, 42, std::ref(l));
        l.arrive_and_wait();

        PIKA_TEST_EQ(annotation, desc);
        PIKA_TEST_EQ(annotation,
            std::string(pika::execution::experimental::get_annotation(exec)));
    }

    {
        annotation.clear();
        auto exec =
            pika::execution::experimental::with_annotation(executor, desc);

        pika::lcos::local::latch l(2);
        pika::parallel::execution::post(exec, &test_post_f, 42, std::ref(l));
        l.arrive_and_wait();

        PIKA_TEST_EQ(annotation, desc);
        PIKA_TEST_EQ(annotation,
            std::string(pika::execution::experimental::get_annotation(exec)));
    }
}

///////////////////////////////////////////////////////////////////////////////
void test(int passed_through)
{
    PIKA_TEST_EQ(passed_through, 42);

    annotation =
        pika::threads::get_thread_description(pika::threads::get_self_id())
            .get_description();
}

template <typename Executor>
void test_sync(Executor&& executor)
{
    std::string desc("test_sync");
    {
        auto exec = pika::experimental::prefer(
            pika::execution::experimental::with_annotation, executor, desc);

        pika::parallel::execution::sync_execute(exec, &test, 42);

        PIKA_TEST_EQ(annotation, desc);
        PIKA_TEST_EQ(annotation,
            std::string(pika::execution::experimental::get_annotation(exec)));
    }

    {
        annotation.clear();
        auto exec =
            pika::execution::experimental::with_annotation(executor, desc);

        pika::parallel::execution::sync_execute(exec, &test, 42);

        PIKA_TEST_EQ(annotation, desc);
        PIKA_TEST_EQ(annotation,
            std::string(pika::execution::experimental::get_annotation(exec)));
    }
}

template <typename Executor>
void test_async(Executor&& executor)
{
    std::string desc("test_async");
    {
        auto exec = pika::experimental::prefer(
            pika::execution::experimental::with_annotation, executor, desc);

        pika::parallel::execution::async_execute(exec, &test, 42).get();

        PIKA_TEST_EQ(annotation, desc);
        PIKA_TEST_EQ(annotation,
            std::string(pika::execution::experimental::get_annotation(exec)));
    }

    {
        annotation.clear();
        auto exec =
            pika::execution::experimental::with_annotation(executor, desc);

        pika::parallel::execution::async_execute(exec, &test, 42).get();

        PIKA_TEST_EQ(annotation, desc);
        PIKA_TEST_EQ(annotation,
            std::string(pika::execution::experimental::get_annotation(exec)));
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_f(pika::future<void> f, int passed_through)
{
    PIKA_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    PIKA_TEST_EQ(passed_through, 42);

    annotation =
        pika::threads::get_thread_description(pika::threads::get_self_id())
            .get_description();
}

template <typename Executor>
void test_then(Executor&& executor)
{
    pika::future<void> f = pika::make_ready_future();

    std::string desc("test_then");
    {
        auto exec = pika::experimental::prefer(
            pika::execution::experimental::with_annotation, executor, desc);

        pika::parallel::execution::then_execute(exec, &test_f, f, 42).get();

        PIKA_TEST_EQ(annotation, desc);
        PIKA_TEST_EQ(annotation,
            std::string(pika::execution::experimental::get_annotation(exec)));
    }

    {
        annotation.clear();
        auto exec =
            pika::execution::experimental::with_annotation(executor, desc);

        pika::parallel::execution::then_execute(exec, &test_f, f, 42).get();

        PIKA_TEST_EQ(annotation, desc);
        PIKA_TEST_EQ(annotation,
            std::string(pika::execution::experimental::get_annotation(exec)));
    }
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test(int seq, int passed_through)    //-V813
{
    PIKA_TEST_EQ(passed_through, 42);

    if (seq == 0)
    {
        annotation =
            pika::threads::get_thread_description(pika::threads::get_self_id())
                .get_description();
    }
}

template <typename Executor>
void test_bulk_sync(Executor&& executor)
{
    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    std::string desc("test_bulk_sync");
    {
        auto exec = pika::experimental::prefer(
            pika::execution::experimental::with_annotation, executor, desc);

        pika::parallel::execution::bulk_sync_execute(
            exec, pika::util::bind(&bulk_test, _1, _2), 107, 42);

        PIKA_TEST_EQ(annotation, desc);
        PIKA_TEST_EQ(annotation,
            std::string(pika::execution::experimental::get_annotation(exec)));

        annotation.clear();
        pika::parallel::execution::bulk_sync_execute(exec, &bulk_test, 107, 42);

        PIKA_TEST_EQ(annotation, desc);
        PIKA_TEST_EQ(annotation,
            std::string(pika::execution::experimental::get_annotation(exec)));
    }

    {
        auto exec =
            pika::execution::experimental::with_annotation(executor, desc);

        annotation.clear();
        pika::parallel::execution::bulk_sync_execute(
            exec, pika::util::bind(&bulk_test, _1, _2), 107, 42);

        PIKA_TEST_EQ(annotation, desc);
        PIKA_TEST_EQ(annotation,
            std::string(pika::execution::experimental::get_annotation(exec)));

        annotation.clear();
        pika::parallel::execution::bulk_sync_execute(exec, &bulk_test, 107, 42);

        PIKA_TEST_EQ(annotation, desc);
        PIKA_TEST_EQ(annotation,
            std::string(pika::execution::experimental::get_annotation(exec)));
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_bulk_async(Executor&& executor)
{
    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    std::string desc("test_bulk_async");
    {
        auto exec = pika::experimental::prefer(
            pika::execution::experimental::with_annotation, executor, desc);

        pika::when_all(pika::parallel::execution::bulk_async_execute(
                          exec, pika::util::bind(&bulk_test, _1, _2), 107, 42))
            .get();

        PIKA_TEST_EQ(annotation, desc);
        PIKA_TEST_EQ(annotation,
            std::string(pika::execution::experimental::get_annotation(exec)));

        annotation.clear();
        pika::when_all(pika::parallel::execution::bulk_async_execute(
                          exec, &bulk_test, 107, 42))
            .get();

        PIKA_TEST_EQ(annotation, desc);
        PIKA_TEST_EQ(annotation,
            std::string(pika::execution::experimental::get_annotation(exec)));
    }

    {
        auto exec =
            pika::execution::experimental::with_annotation(executor, desc);

        annotation.clear();
        pika::when_all(pika::parallel::execution::bulk_async_execute(
                          exec, pika::util::bind(&bulk_test, _1, _2), 107, 42))
            .get();

        PIKA_TEST_EQ(annotation, desc);
        PIKA_TEST_EQ(annotation,
            std::string(pika::execution::experimental::get_annotation(exec)));

        annotation.clear();
        pika::when_all(pika::parallel::execution::bulk_async_execute(
                          exec, &bulk_test, 107, 42))
            .get();

        PIKA_TEST_EQ(annotation, desc);
        PIKA_TEST_EQ(annotation,
            std::string(pika::execution::experimental::get_annotation(exec)));
    }
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test_f(int seq, pika::shared_future<void> f,
    int passed_through)    //-V813
{
    PIKA_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    PIKA_TEST_EQ(passed_through, 42);

    if (seq == 0)
    {
        annotation =
            pika::threads::get_thread_description(pika::threads::get_self_id())
                .get_description();
    }
}

template <typename Executor>
void test_bulk_then(Executor&& executor)
{
    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;
    using pika::util::placeholders::_3;

    pika::shared_future<void> f = pika::make_ready_future();

    std::string desc("test_bulk_then");
    {
        auto exec = pika::experimental::prefer(
            pika::execution::experimental::with_annotation, executor, desc);

        pika::parallel::execution::bulk_then_execute(
            exec, pika::util::bind(&bulk_test_f, _1, _2, _3), 107, f, 42)
            .get();

        PIKA_TEST_EQ(annotation, desc);
        PIKA_TEST_EQ(annotation,
            std::string(pika::execution::experimental::get_annotation(exec)));

        annotation.clear();
        pika::parallel::execution::bulk_then_execute(
            exec, &bulk_test_f, 107, f, 42)
            .get();

        PIKA_TEST_EQ(annotation, desc);
        PIKA_TEST_EQ(annotation,
            std::string(pika::execution::experimental::get_annotation(exec)));
    }

    {
        auto exec =
            pika::execution::experimental::with_annotation(executor, desc);

        annotation.clear();
        pika::parallel::execution::bulk_then_execute(
            exec, pika::util::bind(&bulk_test_f, _1, _2, _3), 107, f, 42)
            .get();

        PIKA_TEST_EQ(annotation, desc);
        PIKA_TEST_EQ(annotation,
            std::string(pika::execution::experimental::get_annotation(exec)));

        annotation.clear();
        pika::parallel::execution::bulk_then_execute(
            exec, &bulk_test_f, 107, f, 42)
            .get();

        PIKA_TEST_EQ(annotation, desc);
        PIKA_TEST_EQ(annotation,
            std::string(pika::execution::experimental::get_annotation(exec)));
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_post_policy(ExPolicy&& policy)
{
    std::string desc("test_post_policy");
    auto p = pika::execution::experimental::with_annotation(policy, desc);

    pika::lcos::local::latch l(2);
    pika::parallel::execution::post(p.executor(), &test_post_f, 42, std::ref(l));
    l.arrive_and_wait();

    PIKA_TEST_EQ(annotation, desc);
    PIKA_TEST_EQ(annotation,
        std::string(pika::execution::experimental::get_annotation(p)));
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_with_annotation(ExPolicy&& policy)
{
    test_post(policy.executor());

    test_sync(policy.executor());
    test_async(policy.executor());
    test_then(policy.executor());

    test_bulk_sync(policy.executor());
    test_bulk_async(policy.executor());
    test_bulk_then(policy.executor());

    test_post_policy(policy);
}

///////////////////////////////////////////////////////////////////////////////
void test_seq_policy()
{
    // make sure execution::seq is not used directly
    {
        auto policy = pika::execution::experimental::with_annotation(
            pika::execution::seq, "seq");

        static_assert(
            !std::is_same<std::decay_t<decltype(policy.executor())>,
                std::decay_t<decltype(pika::execution::seq.executor())>>::value,
            "sequenced_executor should be wrapped in annotating_executor");
    }

    {
        auto original_policy = pika::execution::seq;
        auto policy = pika::execution::experimental::with_annotation(
            std::move(original_policy), "seq");

        static_assert(
            !std::is_same<std::decay_t<decltype(policy.executor())>,
                std::decay_t<decltype(pika::execution::seq.executor())>>::value,
            "sequenced_executor should be wrapped in annotating_executor");
    }
}

void test_par_policy()
{
    // make sure execution::par is used directly
    {
        auto policy = pika::execution::experimental::with_annotation(
            pika::execution::par, "par");

        static_assert(
            std::is_same<std::decay_t<decltype(policy.executor())>,
                std::decay_t<decltype(pika::execution::par.executor())>>::value,
            "parallel_executor should not be wrapped in annotating_executor");
    }

    {
        auto original_policy = pika::execution::par;
        auto policy = pika::execution::experimental::with_annotation(
            std::move(original_policy), "par");

        static_assert(
            std::is_same<std::decay_t<decltype(policy.executor())>,
                std::decay_t<decltype(pika::execution::par.executor())>>::value,
            "parallel_executor should not be wrapped in annotating_executor");
    }
}

///////////////////////////////////////////////////////////////////////////////
struct test_async_executor
{
    using execution_category = pika::execution::parallel_execution_tag;

    template <typename F, typename... Ts>
    static auto async_execute(F&& f, Ts&&... ts)
    {
        return pika::async(std::forward<F>(f), std::forward<Ts>(ts)...);
    }
};

namespace pika { namespace parallel { namespace execution {
    template <>
    struct is_two_way_executor<test_async_executor> : std::true_type
    {
    };
}}}    // namespace pika::parallel::execution

int pika_main()
{
    // supports annotations
    test_with_annotation(pika::execution::par);

    // don't support them
    test_with_annotation(pika::execution::seq);
    test_with_annotation(pika::execution::par.on(test_async_executor()));

    test_seq_policy();
    test_par_policy();

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
