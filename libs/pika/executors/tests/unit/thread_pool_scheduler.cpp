//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/condition_variable.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/mutex.hpp>
#include <pika/testing.hpp>
#include <pika/thread.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

struct custom_type_non_default_constructible_non_copyable
{
    int x;
    custom_type_non_default_constructible_non_copyable() = delete;
    explicit custom_type_non_default_constructible_non_copyable(int x)
      : x(x){};
    custom_type_non_default_constructible_non_copyable(
        custom_type_non_default_constructible_non_copyable&&) = default;
    custom_type_non_default_constructible_non_copyable& operator=(
        custom_type_non_default_constructible_non_copyable&&) = default;
    custom_type_non_default_constructible_non_copyable(
        custom_type_non_default_constructible_non_copyable const&) = delete;
    custom_type_non_default_constructible_non_copyable& operator=(
        custom_type_non_default_constructible_non_copyable const&) = delete;
};

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

///////////////////////////////////////////////////////////////////////////////
void test_execute()
{
    pika::thread::id parent_id = pika::this_thread::get_id();

    ex::thread_pool_scheduler sched{};
    ex::execute(sched, [parent_id]() { PIKA_TEST_NEQ(pika::this_thread::get_id(), parent_id); });
}

struct check_context_receiver
{
    PIKA_STDEXEC_RECEIVER_CONCEPT

    pika::thread::id parent_id;
    pika::mutex& mtx;
    pika::condition_variable& cond;
    bool& executed;

    template <typename E>
    friend void tag_invoke(ex::set_error_t, check_context_receiver&&, E&&) noexcept
    {
        PIKA_TEST(false);
    }

    friend void tag_invoke(ex::set_stopped_t, check_context_receiver&&) noexcept
    {
        PIKA_TEST(false);
    }

    template <typename... Ts>
    friend void tag_invoke(ex::set_value_t, check_context_receiver&& r, Ts&&...) noexcept
    {
        PIKA_TEST_NEQ(r.parent_id, pika::this_thread::get_id());
        PIKA_TEST_NEQ(pika::thread::id(pika::threads::detail::invalid_thread_id),
            pika::this_thread::get_id());
        std::lock_guard l{r.mtx};
        r.executed = true;
        r.cond.notify_one();
    }

    friend constexpr pika::execution::experimental::empty_env tag_invoke(
        pika::execution::experimental::get_env_t, check_context_receiver const&) noexcept
    {
        return {};
    }
};

void test_sender_receiver_basic()
{
    pika::thread::id parent_id = pika::this_thread::get_id();
    pika::mutex mtx;
    pika::condition_variable cond;
    bool executed{false};

    ex::thread_pool_scheduler sched{};

    auto begin = ex::schedule(sched);
    auto os = ex::connect(std::move(begin), check_context_receiver{parent_id, mtx, cond, executed});
    ex::start(os);

    {
        std::unique_lock l{mtx};
        cond.wait(l, [&]() { return executed; });
    }

    PIKA_TEST(executed);
}

pika::thread::id sender_receiver_then_thread_id;

void test_sender_receiver_then()
{
    ex::thread_pool_scheduler sched{};
    pika::thread::id parent_id = pika::this_thread::get_id();
    pika::mutex mtx;
    pika::condition_variable cond;
    bool executed{false};

    auto begin = ex::schedule(sched);
    auto work1 = ex::then(std::move(begin), [=]() {
        sender_receiver_then_thread_id = pika::this_thread::get_id();
        PIKA_TEST_NEQ(sender_receiver_then_thread_id, parent_id);
    });
    auto work2 = ex::then(std::move(work1),
        []() { PIKA_TEST_EQ(sender_receiver_then_thread_id, pika::this_thread::get_id()); });
    auto os = ex::connect(std::move(work2), check_context_receiver{parent_id, mtx, cond, executed});
    ex::start(os);

    {
        std::unique_lock l{mtx};
        cond.wait(l, [&]() { return executed; });
    }

    PIKA_TEST(executed);
}

void test_sender_receiver_then_wait()
{
    ex::thread_pool_scheduler sched{};
    pika::thread::id parent_id = pika::this_thread::get_id();
    std::atomic<std::size_t> then_count{0};
    bool executed{false};

    auto begin = ex::schedule(sched);
    auto work1 = ex::then(std::move(begin), [&then_count, parent_id]() {
        sender_receiver_then_thread_id = pika::this_thread::get_id();
        PIKA_TEST_NEQ(sender_receiver_then_thread_id, parent_id);
        ++then_count;
    });
    auto work2 = ex::then(std::move(work1), [&then_count, &executed]() {
        PIKA_TEST_EQ(sender_receiver_then_thread_id, pika::this_thread::get_id());
        ++then_count;
        executed = true;
    });
    tt::sync_wait(std::move(work2));

    PIKA_TEST_EQ(then_count.load(), std::size_t(2));
    PIKA_TEST(executed);
}

void test_sender_receiver_then_sync_wait()
{
    ex::thread_pool_scheduler sched{};
    pika::thread::id parent_id = pika::this_thread::get_id();
    std::atomic<std::size_t> then_count{0};

    auto begin = ex::schedule(sched);
    auto work = ex::then(std::move(begin), [&then_count, parent_id]() {
        sender_receiver_then_thread_id = pika::this_thread::get_id();
        PIKA_TEST_NEQ(sender_receiver_then_thread_id, parent_id);
        ++then_count;
        return 42;
    });
    auto result = tt::sync_wait(std::move(work));
    PIKA_TEST_EQ(then_count.load(), std::size_t(1));
    static_assert(
        std::is_same<int, std::decay_t<decltype(result)>>::value, "result should be an int");
    PIKA_TEST_EQ(result, 42);
}

void test_sender_receiver_then_arguments()
{
    ex::thread_pool_scheduler sched{};
    pika::thread::id parent_id = pika::this_thread::get_id();
    std::atomic<std::size_t> then_count{0};

    auto begin = ex::schedule(sched);
    auto work1 = ex::then(std::move(begin), [&then_count, parent_id]() {
        sender_receiver_then_thread_id = pika::this_thread::get_id();
        PIKA_TEST_NEQ(sender_receiver_then_thread_id, parent_id);
        ++then_count;
        return 3;
    });
    auto work2 = ex::then(std::move(work1), [&then_count](int x) -> std::string {
        PIKA_TEST_EQ(sender_receiver_then_thread_id, pika::this_thread::get_id());
        ++then_count;
        return std::string("hello") + std::to_string(x);
    });
    auto work3 = ex::then(std::move(work2), [&then_count](std::string s) {
        PIKA_TEST_EQ(sender_receiver_then_thread_id, pika::this_thread::get_id());
        ++then_count;
        return 2 * s.size();
    });
    auto result = tt::sync_wait(std::move(work3));
    PIKA_TEST_EQ(then_count.load(), std::size_t(3));
    static_assert(std::is_same<std::size_t, std::decay_t<decltype(result)>>::value,
        "result should be a std::size_t");
    PIKA_TEST_EQ(result, std::size_t(12));
}

template <typename F>
struct callback_receiver
{
    PIKA_STDEXEC_RECEIVER_CONCEPT

    std::decay_t<F> f;
    pika::mutex& mtx;
    pika::condition_variable& cond;
    bool& executed;

    template <typename E>
    friend void tag_invoke(ex::set_error_t, callback_receiver&&, E&&) noexcept
    {
        PIKA_TEST(false);
    }

    friend void tag_invoke(ex::set_stopped_t, callback_receiver&&) noexcept { PIKA_TEST(false); }

    template <typename... Ts>
    friend void tag_invoke(ex::set_value_t, callback_receiver&& r, Ts&&...) noexcept
    {
        r.f();
        std::lock_guard l{r.mtx};
        r.executed = true;
        r.cond.notify_one();
    }

    friend constexpr pika::execution::experimental::empty_env tag_invoke(
        pika::execution::experimental::get_env_t, callback_receiver const&) noexcept
    {
        return {};
    }
};

void test_properties()
{
    ex::thread_pool_scheduler sched{};
    pika::mutex mtx;
    pika::condition_variable cond;
    bool executed{false};

    constexpr std::array<pika::execution::thread_priority, 3> priorities{
        {pika::execution::thread_priority::low, pika::execution::thread_priority::normal,
            pika::execution::thread_priority::high}};

    for (auto const prio : priorities)
    {
        auto exec_prop = ex::with_priority(sched, prio);
        PIKA_TEST_EQ(ex::get_priority(exec_prop), prio);

        auto check = [prio]() { PIKA_TEST_EQ(prio, pika::this_thread::get_priority()); };
        executed = false;
        auto os = ex::connect(ex::schedule(exec_prop),
            callback_receiver<decltype(check)>{check, mtx, cond, executed});
        ex::start(os);
        {
            std::unique_lock l{mtx};
            cond.wait(l, [&]() { return executed; });
        }

        PIKA_TEST(executed);
    }

    constexpr std::array<pika::execution::thread_stacksize, 4> stacksizes{
        {pika::execution::thread_stacksize::small_, pika::execution::thread_stacksize::medium,
            pika::execution::thread_stacksize::large, pika::execution::thread_stacksize::huge}};

    for (auto const stacksize : stacksizes)
    {
        auto exec_prop = ex::with_stacksize(sched, stacksize);
        PIKA_TEST_EQ(ex::get_stacksize(exec_prop), stacksize);

        auto check = [stacksize]() {
            PIKA_TEST_EQ(stacksize,
                pika::threads::detail::get_thread_id_data(pika::threads::detail::get_self_id())
                    ->get_stack_size_enum());
        };
        executed = false;
        auto os = ex::connect(ex::schedule(exec_prop),
            callback_receiver<decltype(check)>{check, mtx, cond, executed});
        ex::start(os);
        {
            std::unique_lock l{mtx};
            cond.wait(l, [&]() { return executed; });
        }

        PIKA_TEST(executed);
    }

    constexpr std::array<pika::execution::thread_schedule_hint, 4> hints{
        {pika::execution::thread_schedule_hint{}, pika::execution::thread_schedule_hint{1},
            pika::execution::thread_schedule_hint{
                pika::execution::thread_schedule_hint_mode::thread, 2},
            pika::execution::thread_schedule_hint{
                pika::execution::thread_schedule_hint_mode::numa, 3}}};

    for (auto const hint : hints)
    {
        auto exec_prop = ex::with_hint(sched, hint);
        PIKA_TEST(ex::get_hint(exec_prop) == hint);

        // A hint is not guaranteed to be respected, so we only check that the
        // thread_pool_scheduler holds the property.
    }

    {
        char const* annotation = "<test>";
        auto exec_prop = ex::with_annotation(sched, annotation);
        PIKA_TEST_EQ(std::string(ex::get_annotation(exec_prop)), std::string(annotation));

        auto check = [annotation]() {
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
            PIKA_TEST_EQ(std::string(annotation),
                pika::threads::detail::get_thread_description(pika::threads::detail::get_self_id())
                    .get_description());
#else
            (void) annotation;
#endif
        };
        executed = false;
        auto os = ex::connect(ex::schedule(exec_prop),
            callback_receiver<decltype(check)>{check, mtx, cond, executed});
        ex::start(os);

        {
            std::unique_lock l{mtx};
            cond.wait(l, [&]() { return executed; });
        }

        PIKA_TEST(executed);
    }
}

void test_transfer_basic()
{
    ex::thread_pool_scheduler sched{};
    pika::thread::id parent_id = pika::this_thread::get_id();
    pika::thread::id current_id;

    auto begin = ex::schedule(sched);
    auto work1 = ex::then(begin, [=, &current_id]() {
        current_id = pika::this_thread::get_id();
        PIKA_TEST_NEQ(current_id, parent_id);
    });
    auto work2 = ex::then(
        work1, [=, &current_id]() { PIKA_TEST_EQ(current_id, pika::this_thread::get_id()); });
    auto transfer1 = ex::transfer(work2, sched);
    auto work3 = ex::then(transfer1, [=, &current_id]() {
        current_id = pika::this_thread::get_id();
        PIKA_TEST_NEQ(current_id, parent_id);
    });
    auto work4 = ex::then(
        work3, [=, &current_id]() { PIKA_TEST_EQ(current_id, pika::this_thread::get_id()); });
    auto transfer2 = ex::transfer(work4, sched);
    auto work5 = ex::then(transfer2, [=, &current_id]() {
        current_id = pika::this_thread::get_id();
        PIKA_TEST_NEQ(current_id, parent_id);
    });

    tt::sync_wait(std::move(work5));
}

void test_transfer_arguments()
{
    ex::thread_pool_scheduler sched{};
    pika::thread::id parent_id = pika::this_thread::get_id();
    pika::thread::id current_id;

    auto begin = ex::schedule(sched);
    auto work1 = ex::then(begin, [=, &current_id]() {
        current_id = pika::this_thread::get_id();
        PIKA_TEST_NEQ(current_id, parent_id);
        return 3;
    });
    auto work2 = ex::then(work1, [=, &current_id](int x) {
        PIKA_TEST_EQ(current_id, pika::this_thread::get_id());
        return x / 2.0;
    });
    auto transfer1 = ex::transfer(work2, sched);
    auto work3 = ex::then(transfer1, [=, &current_id](double x) {
        current_id = pika::this_thread::get_id();
        PIKA_TEST_NEQ(current_id, parent_id);
        return x / 2;
    });
    auto work4 = ex::then(work3, [=, &current_id](int x) {
        PIKA_TEST_EQ(current_id, pika::this_thread::get_id());
        return "result: " + std::to_string(x);
    });
    auto transfer2 = ex::transfer(work4, sched);
    auto work5 = ex::then(transfer2, [=, &current_id](std::string s) {
        current_id = pika::this_thread::get_id();
        PIKA_TEST_NEQ(current_id, parent_id);
        return s + "!";
    });

    auto result = tt::sync_wait(std::move(work5));
    static_assert(std::is_same<std::string, std::decay_t<decltype(result)>>::value,
        "result should be a std::string");
    PIKA_TEST_EQ(result, std::string("result: 0!"));
}

void test_just_void()
{
    {
        pika::thread::id parent_id = pika::this_thread::get_id();

        auto begin = ex::just();
        auto work1 = ex::then(
            begin, [parent_id]() { PIKA_TEST_EQ(parent_id, pika::this_thread::get_id()); });
        tt::sync_wait(std::move(work1));
    }

    {
        pika::thread::id parent_id = pika::this_thread::get_id();

        auto begin = ex::just();
        auto transfer1 = ex::transfer(begin, ex::thread_pool_scheduler{});
        auto work1 = ex::then(
            transfer1, [parent_id]() { PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id()); });
        tt::sync_wait(std::move(work1));
    }
}

void test_just_one_arg()
{
    {
        pika::thread::id parent_id = pika::this_thread::get_id();

        auto begin = ex::just(3);
        auto work1 = ex::then(begin, [parent_id](int x) {
            PIKA_TEST_EQ(parent_id, pika::this_thread::get_id());
            PIKA_TEST_EQ(x, 3);
        });
        tt::sync_wait(std::move(work1));
    }

    {
        pika::thread::id parent_id = pika::this_thread::get_id();

        auto begin = ex::just(3);
        auto transfer1 = ex::transfer(begin, ex::thread_pool_scheduler{});
        auto work1 = ex::then(transfer1, [parent_id](int x) {
            PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
            PIKA_TEST_EQ(x, 3);
        });
        tt::sync_wait(std::move(work1));
    }
}

void test_just_two_args()
{
    {
        pika::thread::id parent_id = pika::this_thread::get_id();

        auto begin = ex::just(3, std::string("hello"));
        auto work1 = ex::then(begin, [parent_id](int x, std::string y) {
            PIKA_TEST_EQ(parent_id, pika::this_thread::get_id());
            PIKA_TEST_EQ(x, 3);
            PIKA_TEST_EQ(y, std::string("hello"));
        });
        tt::sync_wait(std::move(work1));
    }

    {
        pika::thread::id parent_id = pika::this_thread::get_id();

        auto begin = ex::just(3, std::string("hello"));
        auto transfer1 = ex::transfer(begin, ex::thread_pool_scheduler{});
        auto work1 = ex::then(transfer1, [parent_id](int x, std::string y) {
            PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
            PIKA_TEST_EQ(x, 3);
            PIKA_TEST_EQ(y, std::string("hello"));
        });
        tt::sync_wait(std::move(work1));
    }
}

void test_transfer_just_void()
{
    pika::thread::id parent_id = pika::this_thread::get_id();

    auto begin = ex::transfer_just(ex::thread_pool_scheduler{});
    auto work1 =
        ex::then(begin, [parent_id]() { PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id()); });
    tt::sync_wait(std::move(work1));
}

void test_transfer_just_one_arg()
{
    pika::thread::id parent_id = pika::this_thread::get_id();

    auto begin = ex::transfer_just(ex::thread_pool_scheduler{}, 3);
    auto work1 = ex::then(begin, [parent_id](int x) {
        PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
        PIKA_TEST_EQ(x, 3);
    });
    tt::sync_wait(std::move(work1));
}

void test_transfer_just_two_args()
{
    pika::thread::id parent_id = pika::this_thread::get_id();

    auto begin = ex::transfer_just(ex::thread_pool_scheduler{}, 3, std::string("hello"));
    auto work1 = ex::then(begin, [parent_id](int x, std::string y) {
        PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
        PIKA_TEST_EQ(x, 3);
        PIKA_TEST_EQ(y, std::string("hello"));
    });
    tt::sync_wait(std::move(work1));
}

void test_when_all()
{
    ex::thread_pool_scheduler sched{};

    {
        pika::thread::id parent_id = pika::this_thread::get_id();

        auto work1 = ex::schedule(sched) | ex::then([parent_id]() {
            PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
            return 42;
        });

        auto work2 = ex::schedule(sched) | ex::then([parent_id]() {
            PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
            return std::string("hello");
        });

        auto work3 = ex::schedule(sched) | ex::then([parent_id]() {
            PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
            return 3.14;
        });

        auto when1 = ex::when_all(std::move(work1), std::move(work2), std::move(work3));

        bool executed{false};
        tt::sync_wait(
            std::move(when1) | ex::then([parent_id, &executed](int x, std::string y, double z) {
                PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
                PIKA_TEST_EQ(x, 42);
                PIKA_TEST_EQ(y, std::string("hello"));
                PIKA_TEST_EQ(z, 3.14);
                executed = true;
            }));
        PIKA_TEST(executed);
    }

    {
        pika::thread::id parent_id = pika::this_thread::get_id();

        // The exception is likely to be thrown before set_value from the second
        // sender is called because the second sender sleeps.
        auto work1 = ex::schedule(sched) | ex::then([parent_id]() -> int {
            PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
            throw std::runtime_error("error");
        });

        auto work2 = ex::schedule(sched) | ex::then([parent_id]() {
            PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return std::string("hello");
        });

        bool exception_thrown = false;

        try
        {
            tt::sync_wait(ex::when_all(std::move(work1), std::move(work2)) |
                ex::then([](int, std::string) { PIKA_TEST(false); }));
            PIKA_TEST(false);
        }
        catch (std::runtime_error const& e)
        {
            PIKA_TEST_EQ(std::string(e.what()), std::string("error"));
            exception_thrown = true;
        }

        PIKA_TEST(exception_thrown);
    }

    {
        pika::thread::id parent_id = pika::this_thread::get_id();

        // The exception is likely to be thrown after set_value from the second
        // sender is called because the first sender sleeps before throwing.
        auto work1 = ex::schedule(sched) | ex::then([parent_id]() -> int {
            PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            throw std::runtime_error("error");
        });

        auto work2 = ex::schedule(sched) | ex::then([parent_id]() {
            PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
            return std::string("hello");
        });

        bool exception_thrown = false;

        try
        {
            tt::sync_wait(ex::when_all(std::move(work1), std::move(work2)) |
                ex::then([](int, std::string) { PIKA_TEST(false); }));
            PIKA_TEST(false);
        }
        catch (std::runtime_error const& e)
        {
            PIKA_TEST_EQ(std::string(e.what()), std::string("error"));
            exception_thrown = true;
        }

        PIKA_TEST(exception_thrown);
    }
}

void test_when_all_vector()
{
    ex::thread_pool_scheduler sched{};

    {
        pika::thread::id parent_id = pika::this_thread::get_id();

        std::vector<ex::unique_any_sender<double>> senders;

        senders.emplace_back(ex::schedule(sched) | ex::then([parent_id]() {
            PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
            return 42.0;
        }));

        senders.emplace_back(ex::schedule(sched) | ex::then([parent_id]() {
            PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
            return 43.0;
        }));

        senders.emplace_back(ex::schedule(sched) | ex::then([parent_id]() {
            PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
            return 3.14;
        }));

        auto when1 = ex::when_all_vector(std::move(senders));

        bool executed{false};
        tt::sync_wait(std::move(when1) | ex::then([parent_id, &executed](std::vector<double> v) {
            PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
            PIKA_TEST_EQ(v.size(), std::size_t(3));
            PIKA_TEST_EQ(v[0], 42.0);
            PIKA_TEST_EQ(v[1], 43.0);
            PIKA_TEST_EQ(v[2], 3.14);
            executed = true;
        }));
        PIKA_TEST(executed);
    }

    {
        pika::thread::id parent_id = pika::this_thread::get_id();

        std::vector<ex::unique_any_sender<int>> senders;

        // The exception is likely to be thrown before set_value from the second
        // sender is called because the second sender sleeps.
        senders.emplace_back(ex::schedule(sched) | ex::then([parent_id]() -> int {
            PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
            throw std::runtime_error("error");
        }));

        senders.emplace_back(ex::schedule(sched) | ex::then([parent_id]() {
            PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return 43;
        }));

        bool exception_thrown = false;

        try
        {
            tt::sync_wait(ex::when_all_vector(std::move(senders)) |
                ex::then([](std::vector<int>) { PIKA_TEST(false); }));
            PIKA_TEST(false);
        }
        catch (std::runtime_error const& e)
        {
            PIKA_TEST_EQ(std::string(e.what()), std::string("error"));
            exception_thrown = true;
        }

        PIKA_TEST(exception_thrown);
    }

    {
        pika::thread::id parent_id = pika::this_thread::get_id();

        std::vector<ex::unique_any_sender<int>> senders;

        // The exception is likely to be thrown after set_value from the second
        // sender is called because the first sender sleeps before throwing.
        senders.emplace_back(ex::schedule(sched) | ex::then([parent_id]() {
            PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return 42;
        }));

        senders.emplace_back(ex::schedule(sched) | ex::then([parent_id]() -> int {
            PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
            throw std::runtime_error("error");
        }));

        bool exception_thrown = false;

        try
        {
            tt::sync_wait(ex::when_all_vector(std::move(senders)) |
                ex::then([](std::vector<int>) { PIKA_TEST(false); }));
            PIKA_TEST(false);
        }
        catch (std::runtime_error const& e)
        {
            PIKA_TEST_EQ(std::string(e.what()), std::string("error"));
            exception_thrown = true;
        }

        PIKA_TEST(exception_thrown);
    }
}

void test_ensure_started()
{
    ex::thread_pool_scheduler sched{};

    {
        tt::sync_wait(ex::schedule(sched) | ex::ensure_started());
    }

    {
        auto s = ex::transfer_just(sched, 42) | ex::ensure_started();
        PIKA_TEST_EQ(tt::sync_wait(std::move(s)), 42);
    }

    {
        auto s = ex::transfer_just(sched, 42) | ex::ensure_started() | ex::transfer(sched);
        PIKA_TEST_EQ(tt::sync_wait(std::move(s)), 42);
    }

    {
        auto s = ex::transfer_just(sched, 42) | ex::ensure_started() | ex::split();
        PIKA_TEST_EQ(tt::sync_wait(s), 42);
        PIKA_TEST_EQ(tt::sync_wait(s), 42);
        PIKA_TEST_EQ(tt::sync_wait(s), 42);
        PIKA_TEST_EQ(tt::sync_wait(std::move(s)), 42);
    }

    // It's allowed to discard the sender from ensure_started
    {
        ex::schedule(ex::thread_pool_scheduler{}) | ex::ensure_started();
    }
}

void test_ensure_started_when_all()
{
    ex::thread_pool_scheduler sched{};

    {
        std::atomic<std::size_t> first_task_calls{0};
        std::atomic<std::size_t> successor_task_calls{0};
        pika::mutex mtx;
        pika::condition_variable cond;
        bool started{false};
        auto s = ex::schedule(sched) | ex::then([&]() {
            ++first_task_calls;
            std::lock_guard l{mtx};
            started = true;
            cond.notify_one();
        }) | ex::ensure_started() |
            ex::split();
        {
            std::unique_lock l{mtx};
            cond.wait(l, [&]() { return started; });
        }
        auto succ1 = s | ex::then([&]() {
            ++successor_task_calls;
            return 1;
        });
        auto succ2 = s | ex::then([&]() {
            ++successor_task_calls;
            return 2;
        });
        PIKA_TEST_EQ(tt::sync_wait(ex::when_all(succ1, succ2) |
                         ex::then([](int const& x, int const& y) { return x + y; })),
            3);
        PIKA_TEST_EQ(first_task_calls.load(), std::size_t(1));
        PIKA_TEST_EQ(successor_task_calls.load(), std::size_t(2));
    }

    {
        std::atomic<std::size_t> first_task_calls{0};
        std::atomic<std::size_t> successor_task_calls{0};
        pika::mutex mtx;
        pika::condition_variable cond;
        bool started{false};
        auto s = ex::schedule(sched) | ex::then([&]() {
            ++first_task_calls;
            std::lock_guard l{mtx};
            started = true;
            cond.notify_one();
            return 3;
        }) | ex::ensure_started() |
            ex::split();
        {
            std::unique_lock l{mtx};
            cond.wait(l, [&]() { return started; });
        }
        PIKA_TEST_EQ(first_task_calls.load(), std::size_t(1));
        auto succ1 = s | ex::then([&](int const& x) {
            ++successor_task_calls;
            return x + 1;
        });
        auto succ2 = s | ex::then([&](int const& x) {
            ++successor_task_calls;
            return x + 2;
        });
        PIKA_TEST_EQ(tt::sync_wait(ex::when_all(succ1, succ2) |
                         ex::then([](int const& x, int const& y) { return x + y; })),
            9);
        PIKA_TEST_EQ(first_task_calls.load(), std::size_t(1));
        PIKA_TEST_EQ(successor_task_calls.load(), std::size_t(2));
    }

    {
        std::atomic<std::size_t> first_task_calls{0};
        std::atomic<std::size_t> successor_task_calls{0};
        pika::mutex mtx;
        pika::condition_variable cond;
        bool started{false};
        auto s = ex::schedule(sched) | ex::then([&]() {
            ++first_task_calls;
            std::lock_guard l{mtx};
            started = true;
            cond.notify_one();
            return 3;
        }) | ex::ensure_started() |
            ex::split();
        {
            std::unique_lock l{mtx};
            cond.wait(l, [&]() { return started; });
        }
        auto succ1 = s | ex::transfer(sched) | ex::then([&](int const& x) {
            ++successor_task_calls;
            return x + 1;
        });
        auto succ2 = s | ex::transfer(sched) | ex::then([&](int const& x) {
            ++successor_task_calls;
            return x + 2;
        });
        PIKA_TEST_EQ(tt::sync_wait(ex::when_all(succ1, succ2) |
                         ex::then([](int const& x, int const& y) { return x + y; })),
            9);
        PIKA_TEST_EQ(first_task_calls.load(), std::size_t(1));
        PIKA_TEST_EQ(successor_task_calls.load(), std::size_t(2));
    }
}

void test_split()
{
    ex::thread_pool_scheduler sched{};

    {
        tt::sync_wait(ex::schedule(sched) | ex::split());
    }

    {
        auto s = ex::transfer_just(sched, 42) | ex::split();
        PIKA_TEST_EQ(tt::sync_wait(std::move(s)), 42);
    }

    {
        auto s = ex::transfer_just(sched, 42) | ex::split() | ex::transfer(sched);
        PIKA_TEST_EQ(tt::sync_wait(std::move(s)), 42);
    }

    {
        auto s = ex::transfer_just(sched, 42) | ex::split();
        PIKA_TEST_EQ(tt::sync_wait(s), 42);
        PIKA_TEST_EQ(tt::sync_wait(s), 42);
        PIKA_TEST_EQ(tt::sync_wait(s), 42);
        PIKA_TEST_EQ(tt::sync_wait(std::move(s)), 42);
    }
}

void test_split_when_all()
{
    ex::thread_pool_scheduler sched{};

    {
        std::atomic<std::size_t> first_task_calls{0};
        std::atomic<std::size_t> successor_task_calls{0};
        auto s = ex::schedule(sched) | ex::then([&]() { ++first_task_calls; }) | ex::split();
        auto succ1 = s | ex::then([&]() {
            ++successor_task_calls;
            return 1;
        });
        auto succ2 = s | ex::then([&]() {
            ++successor_task_calls;
            return 2;
        });
        PIKA_TEST_EQ(tt::sync_wait(ex::when_all(succ1, succ2) |
                         ex::then([](int const& x, int const& y) { return x + y; })),
            3);
        PIKA_TEST_EQ(first_task_calls.load(), std::size_t(1));
        PIKA_TEST_EQ(successor_task_calls.load(), std::size_t(2));
    }

    {
        std::atomic<std::size_t> first_task_calls{0};
        std::atomic<std::size_t> successor_task_calls{0};
        auto s = ex::schedule(sched) | ex::then([&]() {
            ++first_task_calls;
            return 3;
        }) | ex::split();
        auto succ1 = s | ex::then([&](int const& x) {
            ++successor_task_calls;
            return x + 1;
        });
        auto succ2 = s | ex::then([&](int const& x) {
            ++successor_task_calls;
            return x + 2;
        });
        PIKA_TEST_EQ(tt::sync_wait(ex::when_all(succ1, succ2) |
                         ex::then([](int const& x, int const& y) { return x + y; })),
            9);
        PIKA_TEST_EQ(first_task_calls.load(), std::size_t(1));
        PIKA_TEST_EQ(successor_task_calls.load(), std::size_t(2));
    }

    {
        std::atomic<std::size_t> first_task_calls{0};
        std::atomic<std::size_t> successor_task_calls{0};
        auto s = ex::schedule(sched) | ex::then([&]() {
            ++first_task_calls;
            return 3;
        }) | ex::split();
        auto succ1 = s | ex::transfer(sched) | ex::then([&](int const& x) {
            ++successor_task_calls;
            return x + 1;
        });
        auto succ2 = s | ex::transfer(sched) | ex::then([&](int const& x) {
            ++successor_task_calls;
            return x + 2;
        });
        PIKA_TEST_EQ(tt::sync_wait(ex::when_all(succ1, succ2) |
                         ex::then([](int const& x, int const& y) { return x + y; })),
            9);
        PIKA_TEST_EQ(first_task_calls.load(), std::size_t(1));
        PIKA_TEST_EQ(successor_task_calls.load(), std::size_t(2));
    }
}

void test_let_value()
{
    ex::thread_pool_scheduler sched{};

    // void predecessor
    {
        auto result =
            tt::sync_wait(ex::schedule(sched) | ex::let_value([]() { return ex::just(42); }));
        PIKA_TEST_EQ(result, 42);
    }

    {
        auto result = tt::sync_wait(
            ex::schedule(sched) | ex::let_value([=]() { return ex::transfer_just(sched, 42); }));
        PIKA_TEST_EQ(result, 42);
    }

    {
        auto result = tt::sync_wait(
            ex::just() | ex::let_value([=]() { return ex::transfer_just(sched, 42); }));
        PIKA_TEST_EQ(result, 42);
    }

    // int predecessor, value ignored
    {
        auto result = tt::sync_wait(
            ex::transfer_just(sched, 43) | ex::let_value([](int&) { return ex::just(42); }));
        PIKA_TEST_EQ(result, 42);
    }

    {
        auto result = tt::sync_wait(ex::transfer_just(sched, 43) |
            ex::let_value([=](int&) { return ex::transfer_just(sched, 42); }));
        PIKA_TEST_EQ(result, 42);
    }

    {
        auto result = tt::sync_wait(
            ex::just(43) | ex::let_value([=](int&) { return ex::transfer_just(sched, 42); }));
        PIKA_TEST_EQ(result, 42);
    }

    // int predecessor, value used
    {
        auto result = tt::sync_wait(ex::transfer_just(sched, 43) | ex::let_value([](int& x) {
            return ex::just(42) | ex::then([&](int y) { return x + y; });
        }));
        PIKA_TEST_EQ(result, 85);
    }

    {
        auto result = tt::sync_wait(ex::transfer_just(sched, 43) | ex::let_value([=](int& x) {
            return ex::transfer_just(sched, 42) | ex::then([&](int y) { return x + y; });
        }));
        PIKA_TEST_EQ(result, 85);
    }

    {
        auto result = tt::sync_wait(ex::just(43) | ex::let_value([=](int& x) {
            return ex::transfer_just(sched, 42) | ex::then([&](int y) { return x + y; });
        }));
        PIKA_TEST_EQ(result, 85);
    }

    // predecessor throws, let sender is ignored
    {
        bool exception_thrown = false;

        try
        {
            tt::sync_wait(ex::transfer_just(sched, 43) | ex::then([](int) -> int {
                throw std::runtime_error("error");
            }) | ex::let_value([](int&) {
                PIKA_TEST(false);
                return ex::just(0);
            }));
            PIKA_TEST(false);
        }
        catch (std::runtime_error const& e)
        {
            PIKA_TEST_EQ(std::string(e.what()), std::string("error"));
            exception_thrown = true;
        }

        PIKA_TEST(exception_thrown);
    }
}

void check_exception_ptr_message(std::exception_ptr ep, std::string const& message)
{
    try
    {
        std::rethrow_exception(ep);
    }
    catch (std::runtime_error const& e)
    {
        PIKA_TEST_EQ(std::string(e.what()), message);
        return;
    }

    PIKA_TEST(false);
}

void test_let_error()
{
    ex::thread_pool_scheduler sched{};

    // void predecessor
    {
        std::atomic<bool> called{false};
        tt::sync_wait(ex::schedule(sched) | ex::then([]() { throw std::runtime_error("error"); }) |
            ex::let_error([&called](std::exception_ptr& ep) {
                called = true;
                check_exception_ptr_message(ep, "error");
                return ex::just();
            }));
        PIKA_TEST(called);
    }

    {
        std::atomic<bool> called{false};
        tt::sync_wait(ex::schedule(sched) | ex::then([]() { throw std::runtime_error("error"); }) |
            ex::let_error([=, &called](std::exception_ptr& ep) {
                called = true;
                check_exception_ptr_message(ep, "error");
                return ex::transfer_just(sched);
            }));
        PIKA_TEST(called);
    }

    {
        std::atomic<bool> called{false};
        tt::sync_wait(ex::just() | ex::then([]() { throw std::runtime_error("error"); }) |
            ex::let_error([=, &called](std::exception_ptr& ep) {
                called = true;
                check_exception_ptr_message(ep, "error");
                return ex::transfer_just(sched);
            }));
        PIKA_TEST(called);
    }

    // int predecessor
    {
        auto result = tt::sync_wait(ex::schedule(sched) | ex::then([]() -> int {
            throw std::runtime_error("error");
        }) | ex::let_error([](std::exception_ptr& ep) {
            check_exception_ptr_message(ep, "error");
            return ex::just(42);
        }));
        PIKA_TEST_EQ(result, 42);
    }

    {
        auto result = tt::sync_wait(ex::schedule(sched) | ex::then([]() -> int {
            throw std::runtime_error("error");
        }) | ex::let_error([=](std::exception_ptr& ep) {
            check_exception_ptr_message(ep, "error");
            return ex::transfer_just(sched, 42);
        }));
        PIKA_TEST_EQ(result, 42);
    }

    {
        auto result = tt::sync_wait(ex::just() | ex::then([]() -> int {
            throw std::runtime_error("error");
        }) | ex::let_error([=](std::exception_ptr& ep) {
            check_exception_ptr_message(ep, "error");
            return ex::transfer_just(sched, 42);
        }));
        PIKA_TEST_EQ(result, 42);
    }

    // predecessor doesn't throw, let sender is ignored
    {
        auto result =
            tt::sync_wait(ex::transfer_just(sched, 42) | ex::let_error([](std::exception_ptr) {
                PIKA_TEST(false);
                return ex::just(43);
            }));
        PIKA_TEST_EQ(result, 42);
    }

    {
        auto result =
            tt::sync_wait(ex::transfer_just(sched, 42) | ex::let_error([=](std::exception_ptr) {
                PIKA_TEST(false);
                return ex::transfer_just(sched, 43);
            }));
        PIKA_TEST_EQ(result, 42);
    }

    {
        auto result = tt::sync_wait(ex::just(42) | ex::let_error([=](std::exception_ptr) {
            PIKA_TEST(false);
            return ex::transfer_just(sched, 43);
        }));
        PIKA_TEST_EQ(result, 42);
    }
}

void test_detach()
{
    ex::thread_pool_scheduler sched{};

    {
        bool called = false;
        pika::mutex mtx;
        pika::condition_variable cond;
        auto s = ex::schedule(sched) | ex::then([&]() {
            std::unique_lock l{mtx};
            called = true;
            cond.notify_one();
        });
        ex::start_detached(std::move(s));

        {
            std::unique_lock l{mtx};
            PIKA_TEST(cond.wait_for(l, std::chrono::seconds(1), [&]() { return called; }));
        }
        PIKA_TEST(called);
    }

    // Values passed to set_value are ignored
    {
        bool called = false;
        pika::mutex mtx;
        pika::condition_variable cond;
        auto s = ex::schedule(sched) | ex::then([&]() {
            std::lock_guard l{mtx};
            called = true;
            cond.notify_one();
            return 42;
        });
        ex::start_detached(std::move(s));

        {
            std::unique_lock l{mtx};
            PIKA_TEST(cond.wait_for(l, std::chrono::seconds(1), [&]() { return called; }));
        }
        PIKA_TEST(called);
    }
}

void test_bulk()
{
    std::vector<int> const ns = {0, 1, 10, 43};

    for (int n : ns)
    {
        std::vector<int> v(n, 0);
        pika::thread::id parent_id = pika::this_thread::get_id();

        tt::sync_wait(ex::schedule(ex::thread_pool_scheduler{}) | ex::bulk(n, [&](int i) {
            ++v[i];
            PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
        }));

        for (int i = 0; i < n; ++i) { PIKA_TEST_EQ(v[i], 1); }
    }

    for (auto n : ns)
    {
        std::vector<int> v(n, -1);
        pika::thread::id parent_id = pika::this_thread::get_id();

        auto v_out = tt::sync_wait(ex::transfer_just(ex::thread_pool_scheduler{}, std::move(v)) |
            ex::bulk(n, [&parent_id](int i, std::vector<int>& v) {
                v[i] = i;
                PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
            }));

        for (int i = 0; i < n; ++i) { PIKA_TEST_EQ(v_out[i], i); }
    }

    // l-value reference sender
    for (int n : ns)
    {
        std::vector<int> v(n, 0);
        pika::thread::id parent_id = pika::this_thread::get_id();

        auto s = ex::schedule(ex::thread_pool_scheduler{}) | ex::bulk(n, [&](int i) {
            ++v[i];
            PIKA_TEST_NEQ(parent_id, pika::this_thread::get_id());
        });
        tt::sync_wait(s);

        for (int i = 0; i < n; ++i) { PIKA_TEST_EQ(v[i], 1); }
    }

    // The specification only allows integral shapes
#if !defined(PIKA_HAVE_STDEXEC)
    {
        std::unordered_set<std::string> string_map;
        std::vector<std::string> v = {"hello", "brave", "new", "world"};
        std::vector<std::string> v_ref = v;

        pika::mutex mtx;

        tt::sync_wait(ex::schedule(ex::thread_pool_scheduler{}) |
            ex::bulk(std::move(v), [&](std::string const& s) {
                std::lock_guard lk(mtx);
                string_map.insert(s);
            }));

        for (auto const& s : v_ref) { PIKA_TEST(string_map.find(s) != string_map.end()); }
    }
#endif

    for (auto n : ns)
    {
        int const i_fail = 3;

        std::vector<int> v(n, -1);
        bool const expect_exception = n > i_fail;

        try
        {
            tt::sync_wait(ex::transfer_just(ex::thread_pool_scheduler{}) | ex::bulk(n, [&v](int i) {
                if (i == i_fail) { throw std::runtime_error("error"); }
                v[i] = i;
            }));

            if (expect_exception) { PIKA_TEST(false); }
        }
        catch (std::runtime_error const& e)
        {
            if (!expect_exception) { PIKA_TEST(false); }

            PIKA_TEST_EQ(std::string(e.what()), std::string("error"));
        }

        if (expect_exception) { PIKA_TEST_EQ(v[i_fail], -1); }
        else
        {
            for (int i = 0; i < n; ++i) { PIKA_TEST_EQ(v[i], i); }
        }
    }
}

void test_completion_scheduler()
{
    {
        auto sender = ex::schedule(ex::thread_pool_scheduler{});
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>, ex::thread_pool_scheduler>,
            "the completion scheduler should be a thread_pool_scheduler");
    }

    {
        auto sender = ex::then(ex::schedule(ex::thread_pool_scheduler{}), []() {});
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>, ex::thread_pool_scheduler>,
            "the completion scheduler should be a thread_pool_scheduler");
    }

    {
        auto sender = ex::transfer_just(ex::thread_pool_scheduler{}, 42);
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>, ex::thread_pool_scheduler>,
            "the completion scheduler should be a thread_pool_scheduler");
    }

    {
        auto sender = ex::bulk(ex::schedule(ex::thread_pool_scheduler{}), 10, [](int) {});
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>, ex::thread_pool_scheduler>,
            "the completion scheduler should be a thread_pool_scheduler");
    }

    {
        auto sender = ex::then(
            ex::bulk(ex::transfer_just(ex::thread_pool_scheduler{}, 42), 10, [](int, int) {}),
            [](int) {});
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>, ex::thread_pool_scheduler>,
            "the completion scheduler should be a thread_pool_scheduler");
    }

    {
        auto sender = ex::bulk(
            ex::then(ex::transfer_just(ex::thread_pool_scheduler{}, 42), [](int x) { return x; }),
            10, [](int, int) {});
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>, ex::thread_pool_scheduler>,
            "the completion scheduler should be a thread_pool_scheduler");
    }
}

void test_drop_value()
{
    ex::thread_pool_scheduler sched{};

    {
        tt::sync_wait(ex::drop_value(ex::schedule(sched)));
        static_assert(std::is_void_v<decltype(tt::sync_wait(ex::drop_value(ex::schedule(sched))))>);
    }

    {
        tt::sync_wait(ex::drop_value(ex::transfer_just(sched, 3)));
        static_assert(
            std::is_void_v<decltype(tt::sync_wait(ex::drop_value(ex::transfer_just(sched, 3))))>);
    }

    {
        tt::sync_wait(ex::drop_value(ex::transfer_just(sched, std::string("hello"))));
        static_assert(std::is_void_v<decltype(tt::sync_wait(
                ex::drop_value(ex::transfer_just(sched, std::string("hello")))))>);
    }

    {
        tt::sync_wait(ex::drop_value(
            ex::transfer_just(sched, custom_type_non_default_constructible_non_copyable{0})));
        static_assert(std::is_void_v<decltype(tt::sync_wait(ex::drop_value(ex::transfer_just(
                sched, custom_type_non_default_constructible_non_copyable{0}))))>);
    }

    {
        auto s = ex::drop_value(ex::then(ex::just(), [] { throw std::runtime_error("error"); }));

        bool exception_thrown = false;

        try
        {
            tt::sync_wait(std::move(s));
            PIKA_TEST(false);
        }
        catch (std::runtime_error const& e)
        {
            PIKA_TEST_EQ(std::string(e.what()), std::string("error"));
            exception_thrown = true;
        }

        PIKA_TEST(exception_thrown);
    }
}

void test_split_tuple()
{
    ex::thread_pool_scheduler sched{};

    {
        auto [s] = ex::split_tuple(ex::transfer_just(sched, std::tuple(42)));
        PIKA_TEST_EQ(tt::sync_wait(std::move(s)), 42);
    }

    {
        auto [s1, s2, s3] =
            ex::split_tuple(ex::transfer_just(sched, std::tuple(42, std::string{"hello"}, 3.14)));
        PIKA_TEST_EQ(tt::sync_wait(std::move(s1)), 42);
        PIKA_TEST_EQ(tt::sync_wait(std::move(s2)), std::string{"hello"});
        PIKA_TEST_EQ(tt::sync_wait(std::move(s3)), 3.14);
    }

    {
        auto [s1, s2, s3] =
            ex::split_tuple(ex::transfer_just(sched, std::tuple(42, std::string{"hello"}, 3.14)));
        auto s1_transfer = std::move(s1) | ex::transfer(sched);
        auto s2_transfer = std::move(s2) | ex::transfer(sched);
        auto s3_transfer = std::move(s3) | ex::transfer(sched);
        PIKA_TEST_EQ(tt::sync_wait(std::move(s1_transfer)), 42);
        PIKA_TEST_EQ(tt::sync_wait(std::move(s2_transfer)), std::string{"hello"});
        PIKA_TEST_EQ(tt::sync_wait(std::move(s3_transfer)), 3.14);
    }

    {
        auto [s1, s2, s3] = ex::split_tuple(ex::then(ex::schedule(sched),
            []() -> std::tuple<int, std::string, double> { throw std::runtime_error("error"); }));

        auto check_exception = [](auto&& s) {
            bool exception_thrown = false;

            try
            {
                tt::sync_wait(std::forward<decltype(s)>(s));
                PIKA_TEST(false);
            }
            catch (std::runtime_error const& e)
            {
                PIKA_TEST_EQ(std::string(e.what()), std::string("error"));
                exception_thrown = true;
            }

            PIKA_TEST(exception_thrown);
        };

        check_exception(std::move(s1));
        check_exception(std::move(s2));
        check_exception(std::move(s3));
    }
}

void test_scheduler_queries()
{
    PIKA_TEST(ex::get_forward_progress_guarantee(ex::thread_pool_scheduler{}) ==
        ex::forward_progress_guarantee::weakly_parallel);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    test_execute();
    test_sender_receiver_basic();
    test_sender_receiver_then();
    test_sender_receiver_then_wait();
    test_sender_receiver_then_sync_wait();
    test_sender_receiver_then_arguments();
    test_properties();
    test_transfer_basic();
    test_transfer_arguments();
    test_just_void();
    test_just_one_arg();
    test_just_two_args();
    test_transfer_just_void();
    test_transfer_just_one_arg();
    test_transfer_just_two_args();
    test_when_all();
    test_when_all_vector();
    test_ensure_started();
    test_ensure_started_when_all();
    test_split();
    test_split_when_all();
    test_let_value();
    test_let_error();
    test_detach();
    test_bulk();
    test_drop_value();
    test_split_tuple();
    test_completion_scheduler();
    test_scheduler_queries();

    pika::finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::init(pika_main, argc, argv), 0, "pika main exited with non-zero status");

    return 0;
}
