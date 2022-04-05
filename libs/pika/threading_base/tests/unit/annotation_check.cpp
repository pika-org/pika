//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Test that creates a set of tasks using normal priority, but every
// Nth normal task spawns a set of high priority tasks.
// The test is intended to be used with a task plotting/profiling
// tool to verify that high priority tasks run before low ones.

#include <pika/execution.hpp>
#include <pika/future.hpp>
#include <pika/init.hpp>
#include <pika/testing.hpp>
#include <pika/thread.hpp>

#include <apex_options.hpp>

#include <atomic>
#include <cstddef>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

//
// This test generates a set of tasks with certain names, then checks
// if the names are present in the screen output from apex.
// The tasks are spawned using dataflow, or continuations with different
// launch policies and executors
// To make scanning the output possible, we prefix the names so that
// the alphabetical output from apex can be scanned with a regex to
// check that the expected names are present.
//
// See the CMakeLists
// set_tests_properties( ...  PROPERTIES PASS_REGULAR_EXPRESSION ...)

// --------------------------------------------------------------------------
// dummy function that just triggers a delay that can be seen in task plots
void dummy_task(std::size_t n)
{
    // no other work can take place on this thread whilst it sleeps
    bool sleep = true;
    auto start = std::chrono::steady_clock::now();
    do
    {
        std::this_thread::sleep_for(std::chrono::microseconds(n) / 25);
        auto now = std::chrono::steady_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(now - start);
        sleep = (elapsed < std::chrono::microseconds(n));
    } while (sleep);
}

auto test_senders()
{
    ex::execute(
        ex::with_annotation(ex::thread_pool_scheduler{}, "0-execute"), [] {});

    {
        pika::scoped_annotation ann{"0-execute-parent-annotation"};
        ex::execute(ex::thread_pool_scheduler{}, [] {});
        ex::execute(ex::with_annotation(ex::thread_pool_scheduler{},
                        "0-execute-with-annotation-override"),
            [] {});
        ex::execute(ex::with_annotation(ex::thread_pool_scheduler{},
                        "0-this-should-not-become-an-annotation"),
            pika::annotated_function([] {}, "0-execute-annotated-function"));
    }

    auto s1 = ex::schedule(
        ex::with_annotation(ex::thread_pool_scheduler{}, "0-schedule"));
    auto s2 = ex::schedule(ex::thread_pool_scheduler{}) |
        ex::then(pika::annotated_function(
            [] { dummy_task(1000); }, "0-schedule-then"));
    auto s3 = ex::just() |
        ex::transfer(
            ex::with_annotation(ex::thread_pool_scheduler{}, "0-transfer"));

    return ex::when_all(std::move(s1), std::move(s2), std::move(s3));
}

// --------------------------------------------------------------------------
// string for a policy
std::string policy_string(const pika::launch& policy)
{
    if (policy == pika::launch::async)
    {
        return "async";
    }
    else if (policy == pika::launch::sync)
    {
        return "sync";
    }
    else if (policy == pika::launch::fork)
    {
        return "fork";
    }
    else if (policy == pika::launch::apply)
    {
        return "apply";
    }
    else if (policy == pika::launch::deferred)
    {
        return "deferred";
    }
    else
    {
        return "policy ?";
    }
}

// string for an executor
template <typename Executor>
std::string exec_string(const Executor&)
{
    return "Executor";
}

// --------------------------------------------------------------------------
template <typename Executor>
typename std::enable_if<pika::traits::is_executor_any<Executor>::value,
    std::string>::type
execution_string(const Executor& exec)
{
    return exec_string(exec);
}

template <typename Policy>
typename std::enable_if<pika::detail::is_launch_policy<Policy>::value,
    std::string>::type
execution_string(const Policy& policy)
{
    return policy_string(policy);
}

// --------------------------------------------------------------------------
// use annotate_function
void test_annotate_function()
{
    pika::async([]() {
        pika::scoped_annotation annotate("4-char annotate_function");
    }).get();

    pika::async([]() {
        std::string s("4-string annotate_function");
        pika::scoped_annotation annotate(std::move(s));
    }).get();
}

// --------------------------------------------------------------------------
// no executor or policy
pika::future<void> test_none()
{
    std::string dfs = std::string("1-Dataflow");
    std::string pcs = std::string("2-Continuation");
    std::string pcsu = std::string("3-Unwrapping Continuation");

    std::vector<pika::future<void>> results;
    {
        pika::future<int> f1 = pika::async([]() { return 5; });
        pika::future<int> f2 = pika::make_ready_future(5);
        results.emplace_back(pika::dataflow(
            pika::annotated_function(
                [](auto&&, auto&&) { dummy_task(std::size_t(1000)); }, dfs),
            f1, f2));
    }

    {
        pika::future<int> f1 = pika::async([]() { return 5; });
        results.emplace_back(f1.then(pika::annotated_function(
            [](auto&&) { dummy_task(std::size_t(1000)); }, pcs)));
    }

    {
        pika::future<int> f1 = pika::async([]() { return 5; });
        results.emplace_back(f1.then(pika::unwrapping(pika::annotated_function(
            [](auto&&) { dummy_task(std::size_t(1000)); }, pcsu))));
    }

    // wait for completion
    return pika::when_all(results);
}

// --------------------------------------------------------------------------
// can be called with an executor or a policy
template <typename Execution>
pika::future<void> test_execution(Execution& exec)
{
    static int prefix = 1;
    std::string dfs = std::to_string(prefix++) + "-" + execution_string(exec) +
        std::string(" Dataflow");
    std::string pcs = std::to_string(prefix++) + "-" + execution_string(exec) +
        std::string(" Continuation");
    std::string pcsu = std::to_string(prefix++) + "-" + execution_string(exec) +
        std::string(" Unwrapping Continuation");

    std::vector<pika::future<void>> results;
    {
        pika::future<int> f1 = pika::async([]() { return 5; });
        pika::future<int> f2 = pika::make_ready_future(5);
        results.emplace_back(pika::dataflow(exec,
            pika::annotated_function(
                [](auto&&, auto&&) { dummy_task(std::size_t(1000)); }, dfs),
            f1, f2));
    }
    {
        pika::future<int> f1 = pika::async([]() { return 5; });
        results.emplace_back(f1.then(exec,
            pika::annotated_function(
                [](auto&&) { dummy_task(std::size_t(1000)); }, pcs)));
    }
    {
        pika::future<int> f1 = pika::async([]() { return 5; });
        results.emplace_back(f1.then(exec,
            pika::unwrapping(pika::annotated_function(
                [](auto&&) { dummy_task(std::size_t(1000)); }, pcsu))));
    }
    // wait for completion
    return pika::when_all(results);
}

int pika_main()
{
    tt::sync_wait(test_senders());

    // setup executors
    pika::execution::parallel_executor par_exec{};

    test_annotate_function();
    //
    test_none().get();
    //
    test_execution(pika::launch::apply).get();
    test_execution(pika::launch::async).get();
    test_execution(pika::launch::deferred).get();
    test_execution(pika::launch::fork).get();
    test_execution(pika::launch::sync).get();
    //
    test_execution(par_exec).get();
    //
    return pika::finalize();
}

int main(int argc, char* argv[])
{
    apex::apex_options::use_screen_output(true);
    PIKA_TEST_EQ(pika::init(pika_main, argc, argv), 0);
    return pika::util::report_errors();
}
