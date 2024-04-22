//  Copyright (c) 2020-2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/testing.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
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
    std::thread::id parent_id = std::this_thread::get_id();

    ex::std_thread_scheduler sched{};
    ex::execute(sched, [parent_id]() { PIKA_TEST_NEQ(std::this_thread::get_id(), parent_id); });
}

struct check_context_receiver
{
    PIKA_STDEXEC_RECEIVER_CONCEPT

    std::thread::id parent_id;
    std::mutex& mtx;
    std::condition_variable& cond;
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
        PIKA_TEST_NEQ(r.parent_id, std::this_thread::get_id());
        PIKA_TEST_NEQ(std::thread::id(), std::this_thread::get_id());
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
    std::thread::id parent_id = std::this_thread::get_id();
    std::mutex mtx;
    std::condition_variable cond;
    bool executed{false};

    ex::std_thread_scheduler sched{};

    auto begin = ex::schedule(sched);
    auto os = ex::connect(std::move(begin), check_context_receiver{parent_id, mtx, cond, executed});
    ex::start(os);

    {
        std::unique_lock l{mtx};
        cond.wait(l, [&]() { return executed; });
    }

    PIKA_TEST(executed);
}

std::thread::id sender_receiver_then_thread_id;

void test_sender_receiver_then()
{
    ex::std_thread_scheduler sched{};
    std::thread::id parent_id = std::this_thread::get_id();
    std::mutex mtx;
    std::condition_variable cond;
    bool executed{false};

    auto begin = ex::schedule(sched);
    auto work1 = ex::then(std::move(begin), [=]() {
        sender_receiver_then_thread_id = std::this_thread::get_id();
        PIKA_TEST_NEQ(sender_receiver_then_thread_id, parent_id);
    });
    auto work2 = ex::then(std::move(work1),
        []() { PIKA_TEST_EQ(sender_receiver_then_thread_id, std::this_thread::get_id()); });
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
    ex::std_thread_scheduler sched{};
    std::thread::id parent_id = std::this_thread::get_id();
    std::atomic<std::size_t> then_count{0};
    bool executed{false};

    auto begin = ex::schedule(sched);
    auto work1 = ex::then(std::move(begin), [&then_count, parent_id]() {
        sender_receiver_then_thread_id = std::this_thread::get_id();
        PIKA_TEST_NEQ(sender_receiver_then_thread_id, parent_id);
        ++then_count;
    });
    auto work2 = ex::then(std::move(work1), [&then_count, &executed]() {
        PIKA_TEST_EQ(sender_receiver_then_thread_id, std::this_thread::get_id());
        ++then_count;
        executed = true;
    });
    tt::sync_wait(std::move(work2));

    PIKA_TEST_EQ(then_count.load(), std::size_t(2));
    PIKA_TEST(executed);
}

void test_sender_receiver_then_sync_wait()
{
    ex::std_thread_scheduler sched{};
    std::thread::id parent_id = std::this_thread::get_id();
    std::atomic<std::size_t> then_count{0};

    auto begin = ex::schedule(sched);
    auto work = ex::then(std::move(begin), [&then_count, parent_id]() {
        sender_receiver_then_thread_id = std::this_thread::get_id();
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
    ex::std_thread_scheduler sched{};
    std::thread::id parent_id = std::this_thread::get_id();
    std::atomic<std::size_t> then_count{0};

    auto begin = ex::schedule(sched);
    auto work1 = ex::then(std::move(begin), [&then_count, parent_id]() {
        sender_receiver_then_thread_id = std::this_thread::get_id();
        PIKA_TEST_NEQ(sender_receiver_then_thread_id, parent_id);
        ++then_count;
        return 3;
    });
    auto work2 = ex::then(std::move(work1), [&then_count](int x) -> std::string {
        PIKA_TEST_EQ(sender_receiver_then_thread_id, std::this_thread::get_id());
        ++then_count;
        return std::string("hello") + std::to_string(x);
    });
    auto work3 = ex::then(std::move(work2), [&then_count](std::string s) {
        PIKA_TEST_EQ(sender_receiver_then_thread_id, std::this_thread::get_id());
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
    std::mutex& mtx;
    std::condition_variable& cond;
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

void test_transfer_basic()
{
    ex::std_thread_scheduler sched{};
    std::thread::id parent_id = std::this_thread::get_id();
    std::thread::id current_id;

    auto begin = ex::schedule(sched);
    auto work1 = ex::then(begin, [=, &current_id]() {
        current_id = std::this_thread::get_id();
        PIKA_TEST_NEQ(current_id, parent_id);
    });
    auto work2 = ex::then(std::move(work1),
        [=, &current_id]() { PIKA_TEST_EQ(current_id, std::this_thread::get_id()); });
    auto transfer1 = ex::transfer(work2, sched);
    auto work3 = ex::then(transfer1, [=, &current_id]() {
        std::thread::id new_id = std::this_thread::get_id();
        PIKA_TEST_NEQ(current_id, new_id);
        current_id = new_id;
        PIKA_TEST_NEQ(current_id, parent_id);
    });
    auto work4 = ex::then(
        work3, [=, &current_id]() { PIKA_TEST_EQ(current_id, std::this_thread::get_id()); });
    auto transfer2 = ex::transfer(work4, sched);
    auto work5 = ex::then(transfer2, [=, &current_id]() {
        std::thread::id new_id = std::this_thread::get_id();
        PIKA_TEST_NEQ(current_id, new_id);
        current_id = new_id;
        PIKA_TEST_NEQ(current_id, parent_id);
    });

    tt::sync_wait(std::move(work5));
}

void test_transfer_arguments()
{
    ex::std_thread_scheduler sched{};
    std::thread::id parent_id = std::this_thread::get_id();
    std::thread::id current_id;

    auto begin = ex::schedule(sched);
    auto work1 = ex::then(begin, [=, &current_id]() {
        current_id = std::this_thread::get_id();
        PIKA_TEST_NEQ(current_id, parent_id);
        return 3;
    });
    auto work2 = ex::then(work1, [=, &current_id](int x) {
        PIKA_TEST_EQ(current_id, std::this_thread::get_id());
        return x / 2.0;
    });
    auto transfer1 = ex::transfer(work2, sched);
    auto work3 = ex::then(transfer1, [=, &current_id](double x) {
        std::thread::id new_id = std::this_thread::get_id();
        PIKA_TEST_NEQ(current_id, new_id);
        current_id = new_id;
        PIKA_TEST_NEQ(current_id, parent_id);
        return x / 2;
    });
    auto work4 = ex::then(work3, [=, &current_id](int x) {
        PIKA_TEST_EQ(current_id, std::this_thread::get_id());
        return "result: " + std::to_string(x);
    });
    auto transfer2 = ex::transfer(work4, sched);
    auto work5 = ex::then(transfer2, [=, &current_id](std::string s) {
        std::thread::id new_id = std::this_thread::get_id();
        PIKA_TEST_NEQ(current_id, new_id);
        current_id = new_id;
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
        std::thread::id parent_id = std::this_thread::get_id();

        auto begin = ex::just();
        auto work1 =
            ex::then(begin, [parent_id]() { PIKA_TEST_EQ(parent_id, std::this_thread::get_id()); });
        tt::sync_wait(std::move(work1));
    }

    {
        std::thread::id parent_id = std::this_thread::get_id();

        auto begin = ex::just();
        auto transfer1 = ex::transfer(begin, ex::std_thread_scheduler{});
        auto work1 = ex::then(
            transfer1, [parent_id]() { PIKA_TEST_NEQ(parent_id, std::this_thread::get_id()); });
        tt::sync_wait(std::move(work1));
    }
}

void test_just_one_arg()
{
    {
        std::thread::id parent_id = std::this_thread::get_id();

        auto begin = ex::just(3);
        auto work1 = ex::then(begin, [parent_id](int x) {
            PIKA_TEST_EQ(parent_id, std::this_thread::get_id());
            PIKA_TEST_EQ(x, 3);
        });
        tt::sync_wait(std::move(work1));
    }

    {
        std::thread::id parent_id = std::this_thread::get_id();

        auto begin = ex::just(3);
        auto transfer1 = ex::transfer(begin, ex::std_thread_scheduler{});
        auto work1 = ex::then(transfer1, [parent_id](int x) {
            PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
            PIKA_TEST_EQ(x, 3);
        });
        tt::sync_wait(std::move(work1));
    }
}

void test_just_two_args()
{
    {
        std::thread::id parent_id = std::this_thread::get_id();

        auto begin = ex::just(3, std::string("hello"));
        auto work1 = ex::then(begin, [parent_id](int x, std::string y) {
            PIKA_TEST_EQ(parent_id, std::this_thread::get_id());
            PIKA_TEST_EQ(x, 3);
            PIKA_TEST_EQ(y, std::string("hello"));
        });
        tt::sync_wait(std::move(work1));
    }

    {
        std::thread::id parent_id = std::this_thread::get_id();

        auto begin = ex::just(3, std::string("hello"));
        auto transfer1 = ex::transfer(begin, ex::std_thread_scheduler{});
        auto work1 = ex::then(transfer1, [parent_id](int x, std::string y) {
            PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
            PIKA_TEST_EQ(x, 3);
            PIKA_TEST_EQ(y, std::string("hello"));
        });
        tt::sync_wait(std::move(work1));
    }
}

void test_transfer_just_void()
{
    std::thread::id parent_id = std::this_thread::get_id();

    auto begin = ex::transfer_just(ex::std_thread_scheduler{});
    auto work1 =
        ex::then(begin, [parent_id]() { PIKA_TEST_NEQ(parent_id, std::this_thread::get_id()); });
    tt::sync_wait(std::move(work1));
}

void test_transfer_just_one_arg()
{
    std::thread::id parent_id = std::this_thread::get_id();

    auto begin = ex::transfer_just(ex::std_thread_scheduler{}, 3);
    auto work1 = ex::then(begin, [parent_id](int x) {
        PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
        PIKA_TEST_EQ(x, 3);
    });
    tt::sync_wait(std::move(work1));
}

void test_transfer_just_two_args()
{
    std::thread::id parent_id = std::this_thread::get_id();

    auto begin = ex::transfer_just(ex::std_thread_scheduler{}, 3, std::string("hello"));
    auto work1 = ex::then(begin, [parent_id](int x, std::string y) {
        PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
        PIKA_TEST_EQ(x, 3);
        PIKA_TEST_EQ(y, std::string("hello"));
    });
    tt::sync_wait(std::move(work1));
}

void test_when_all()
{
    ex::std_thread_scheduler sched{};

    {
        std::thread::id parent_id = std::this_thread::get_id();

        auto work1 = ex::schedule(sched) | ex::then([parent_id]() {
            PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
            return 42;
        });

        auto work2 = ex::schedule(sched) | ex::then([parent_id]() {
            PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
            return std::string("hello");
        });

        auto work3 = ex::schedule(sched) | ex::then([parent_id]() {
            PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
            return 3.14;
        });

        auto when1 = ex::when_all(std::move(work1), std::move(work2), std::move(work3));

        bool executed{false};
        tt::sync_wait(
            std::move(when1) | ex::then([parent_id, &executed](int x, std::string y, double z) {
                PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
                PIKA_TEST_EQ(x, 42);
                PIKA_TEST_EQ(y, std::string("hello"));
                PIKA_TEST_EQ(z, 3.14);
                executed = true;
            }));
        PIKA_TEST(executed);
    }

    {
        std::thread::id parent_id = std::this_thread::get_id();

        // The exception is likely to be thrown before set_value from the second
        // sender is called because the second sender sleeps.
        auto work1 = ex::schedule(sched) | ex::then([parent_id]() -> int {
            PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
            throw std::runtime_error("error");
        });

        auto work2 = ex::schedule(sched) | ex::then([parent_id]() {
            PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
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
        std::thread::id parent_id = std::this_thread::get_id();

        // The exception is likely to be thrown after set_value from the second
        // sender is called because the first sender sleeps before throwing.
        auto work1 = ex::schedule(sched) | ex::then([parent_id]() -> int {
            PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            throw std::runtime_error("error");
        });

        auto work2 = ex::schedule(sched) | ex::then([parent_id]() {
            PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
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
    ex::std_thread_scheduler sched{};

    {
        std::thread::id parent_id = std::this_thread::get_id();

        std::vector<ex::unique_any_sender<double>> senders;

        senders.emplace_back(ex::schedule(sched) | ex::then([parent_id]() {
            PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
            return 42.0;
        }));

        senders.emplace_back(ex::schedule(sched) | ex::then([parent_id]() {
            PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
            return 43.0;
        }));

        senders.emplace_back(ex::schedule(sched) | ex::then([parent_id]() {
            PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
            return 3.14;
        }));

        auto when1 = ex::when_all_vector(std::move(senders));

        bool executed{false};
        tt::sync_wait(std::move(when1) | ex::then([parent_id, &executed](std::vector<double> v) {
            PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
            PIKA_TEST_EQ(v.size(), std::size_t(3));
            PIKA_TEST_EQ(v[0], 42.0);
            PIKA_TEST_EQ(v[1], 43.0);
            PIKA_TEST_EQ(v[2], 3.14);
            executed = true;
        }));
        PIKA_TEST(executed);
    }

    {
        std::thread::id parent_id = std::this_thread::get_id();

        std::vector<ex::unique_any_sender<int>> senders;

        // The exception is likely to be thrown before set_value from the second
        // sender is called because the second sender sleeps.
        senders.emplace_back(ex::schedule(sched) | ex::then([parent_id]() -> int {
            PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
            throw std::runtime_error("error");
        }));

        senders.emplace_back(ex::schedule(sched) | ex::then([parent_id]() {
            PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
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
        std::thread::id parent_id = std::this_thread::get_id();

        std::vector<ex::unique_any_sender<int>> senders;

        // The exception is likely to be thrown after set_value from the second
        // sender is called because the first sender sleeps before throwing.
        senders.emplace_back(ex::schedule(sched) | ex::then([parent_id]() {
            PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return 42;
        }));

        senders.emplace_back(ex::schedule(sched) | ex::then([parent_id]() -> int {
            PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
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
    ex::std_thread_scheduler sched{};

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
        ex::schedule(ex::std_thread_scheduler{}) | ex::ensure_started();
    }
}

void test_ensure_started_when_all()
{
    ex::std_thread_scheduler sched{};

    {
        std::atomic<std::size_t> first_task_calls{0};
        std::atomic<std::size_t> successor_task_calls{0};
        std::mutex mtx;
        std::condition_variable cond;
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
        std::mutex mtx;
        std::condition_variable cond;
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
        std::mutex mtx;
        std::condition_variable cond;
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
    ex::std_thread_scheduler sched{};

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
    ex::std_thread_scheduler sched{};

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
    ex::std_thread_scheduler sched{};

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
    ex::std_thread_scheduler sched{};

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
    ex::std_thread_scheduler sched{};

    {
        bool called = false;
        std::mutex mtx;
        std::condition_variable cond;
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
        std::mutex mtx;
        std::condition_variable cond;
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
        std::thread::id parent_id = std::this_thread::get_id();

        tt::sync_wait(ex::schedule(ex::std_thread_scheduler{}) | ex::bulk(n, [&](int i) {
            ++v[i];
            PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
        }));

        for (int i = 0; i < n; ++i) { PIKA_TEST_EQ(v[i], 1); }
    }

    for (auto n : ns)
    {
        std::vector<int> v(n, -1);
        std::thread::id parent_id = std::this_thread::get_id();

        auto v_out = tt::sync_wait(ex::transfer_just(ex::std_thread_scheduler{}, std::move(v)) |
            ex::bulk(n, [&parent_id](int i, std::vector<int>& v) {
                v[i] = i;
                PIKA_TEST_NEQ(parent_id, std::this_thread::get_id());
            }));

        for (int i = 0; i < n; ++i) { PIKA_TEST_EQ(v_out[i], i); }
    }

    // The specification only allows integral shapes
#if !defined(PIKA_HAVE_STDEXEC)
    {
        std::unordered_set<std::string> string_map;
        std::vector<std::string> v = {"hello", "brave", "new", "world"};
        std::vector<std::string> v_ref = v;

        std::mutex mtx;

        tt::sync_wait(ex::schedule(ex::std_thread_scheduler{}) |
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
            tt::sync_wait(ex::transfer_just(ex::std_thread_scheduler{}) | ex::bulk(n, [&v](int i) {
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
        auto sender = ex::schedule(ex::std_thread_scheduler{});
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>, ex::std_thread_scheduler>,
            "the completion scheduler should be a std_thread_scheduler");
    }

    {
        auto sender = ex::then(ex::schedule(ex::std_thread_scheduler{}), []() {});
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>, ex::std_thread_scheduler>,
            "the completion scheduler should be a std_thread_scheduler");
    }

    {
        auto sender = ex::transfer_just(ex::std_thread_scheduler{}, 42);
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>, ex::std_thread_scheduler>,
            "the completion scheduler should be a std_thread_scheduler");
    }

    {
        auto sender = ex::bulk(ex::schedule(ex::std_thread_scheduler{}), 10, [](int) {});
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>, ex::std_thread_scheduler>,
            "the completion scheduler should be a std_thread_scheduler");
    }

    {
        auto sender = ex::then(
            ex::bulk(ex::transfer_just(ex::std_thread_scheduler{}, 42), 10, [](int, int) {}),
            [](int) {});
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>, ex::std_thread_scheduler>,
            "the completion scheduler should be a std_thread_scheduler");
    }

    {
        auto sender =
            ex::bulk(ex::then(ex::transfer_just(ex::std_thread_scheduler{}, 42), [](int) {}), 10,
                [](int, int) {});
        auto completion_scheduler =
            ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
        static_assert(
            std::is_same_v<std::decay_t<decltype(completion_scheduler)>, ex::std_thread_scheduler>,
            "the completion scheduler should be a std_thread_scheduler");
    }
}

void test_scheduler_queries()
{
    PIKA_TEST(ex::get_forward_progress_guarantee(ex::std_thread_scheduler{}) ==
        ex::forward_progress_guarantee::weakly_parallel);
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    test_execute();
    test_sender_receiver_basic();
    test_sender_receiver_then();
    test_sender_receiver_then_wait();
    test_sender_receiver_then_sync_wait();
    test_sender_receiver_then_arguments();
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
    test_completion_scheduler();
    test_scheduler_queries();

    return 0;
}
