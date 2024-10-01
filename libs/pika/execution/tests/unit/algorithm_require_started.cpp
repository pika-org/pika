//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/execution_base/tests/algorithm_test_utils.hpp>
#include <pika/testing.hpp>
#include <pika/type_support/unused.hpp>

#include <atomic>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

void test_basics()
{
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> started{false};
        auto s1 = ex::then(void_sender{}, [&] { started = true; });
        auto s2 = ex::require_started(std::move(s1));
        PIKA_TEST(!started);
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        PIKA_TEST(!started);
        ex::start(os);
        PIKA_TEST(started);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> started{false};
        auto s1 = ex::then(ex::just(0), [&](int x) {
            started = true;
            return x;
        });
        auto s2 = ex::require_started(std::move(s1));
        PIKA_TEST(!started);
        auto f = [](int x) { PIKA_TEST_EQ(x, 0); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        PIKA_TEST(!started);
        ex::start(os);
        PIKA_TEST(started);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> started{false};
        auto s1 = ex::then(ex::just(custom_type_non_default_constructible{42}),
            [&](custom_type_non_default_constructible x) {
                started = true;
                return x;
            });
        auto s2 = ex::require_started(std::move(s1));
        PIKA_TEST(!started);
        auto f = [](auto x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        PIKA_TEST(!started);
        ex::start(os);
        PIKA_TEST(started);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> started{false};
        auto s1 = ex::then(ex::just(custom_type_non_default_constructible_non_copyable{42}),
            [&](custom_type_non_default_constructible_non_copyable&& x) {
                started = true;
                return std::move(x);
            });
        auto s2 = ex::require_started(std::move(s1));
        PIKA_TEST(!started);
        auto f = [](auto&& x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        PIKA_TEST(!started);
        ex::start(os);
        PIKA_TEST(started);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        int x = 42;
        auto s1 = const_reference_sender<int>{x};
        auto s2 = ex::require_started(std::move(s1));
        auto f = [](auto&& x) { PIKA_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }
}

void test_pipe_operator()
{
    {
        std::atomic<bool> set_value_called{false};
        auto s = void_sender{} | ex::require_started();
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }
}

void test_failure()
{
    {
        std::atomic<bool> set_error_called{false};
        auto s = error_sender{} | ex::require_started();
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto s = const_reference_error_sender{} | ex::require_started();
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto s =
            error_sender{} | ex::require_started() | ex::require_started() | ex::require_started();
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }
}

void test_adl()
{
    auto s = ex::require_started(my_namespace::my_sender{});
    test_adl_isolation(s);
    tt::sync_wait(std::move(s));
}

enum class expect_exception
{
    no,
    yes
};

// Check that the sender throws if it's not connected or the operation state isn't started
template <typename F>
void check_exception(expect_exception e, F test)
{
    try
    {
        test();
        PIKA_TEST(e == expect_exception::no);
    }
    catch (...)
    {
        PIKA_TEST(e == expect_exception::yes);
    }
}

void test_no_exception()
{
    // Expect no exception to be thrown
    check_exception(expect_exception::no, [] {
        auto s = void_sender{};
        auto rs = ex::require_started(s);
        tt::sync_wait(std::move(rs));
    });

    check_exception(expect_exception::no, [] {
        auto s = void_sender{};
        auto rs1 = ex::require_started(s);
        auto rs2 = std::move(rs1);
        tt::sync_wait(std::move(rs2));
    });

    check_exception(expect_exception::no, [] {
        auto s = void_sender{};
        auto rs1 = ex::require_started(s);
        auto rs2 = std::move(rs1);
        rs1 = std::move(rs2);
        tt::sync_wait(std::move(rs1));
    });

    check_exception(expect_exception::no, [] {
        auto s = void_sender{};
        auto rs1 = ex::require_started(s);
        auto rs2 = std::move(rs1);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        auto rs3 = std::move(rs1);
        tt::sync_wait(std::move(rs2));
    });

    check_exception(expect_exception::no, [] {
        auto s = void_sender{};
        auto rs1 = ex::require_started(s);
        auto rs2 = std::move(rs1);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        auto rs3 = rs1;
        tt::sync_wait(std::move(rs2));
    });
}

enum class exception_test_mode
{
    no_discard,
    discard
};

template <typename S>
void discard_if_required(S& s, exception_test_mode mode)
{
    if (mode == exception_test_mode::discard) { s.discard(); }
}

void test_exception(exception_test_mode mode)
{
#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
    // When the mode is no_discard we expect exceptions, otherwise we don't expect exceptions
    check_exception(
        mode == exception_test_mode::discard ? expect_exception::no : expect_exception::yes, [=] {
            auto rs = ex::require_started(
                void_sender{}, ex::require_started_mode::throw_on_unstarted);    // never started
            discard_if_required(rs, mode);
        });

    // Move constructor
    check_exception(
        mode == exception_test_mode::discard ? expect_exception::no : expect_exception::yes, [=] {
            auto rs1 = ex::require_started(
                void_sender{}, ex::require_started_mode::throw_on_unstarted);    // empty
            auto rs2 = std::move(rs1);                                           // never started
            discard_if_required(rs2, mode);
        });

    // Copy constructor
    check_exception(
        mode == exception_test_mode::discard ? expect_exception::no : expect_exception::yes, [=] {
            auto rs1 =
                ex::require_started(void_sender{}, ex::require_started_mode::throw_on_unstarted);
            auto rs2 = rs1;        // never started
            tt::sync_wait(rs1);    // started
            discard_if_required(rs2, mode);
        });

    // Move assignment
    check_exception(
        mode == exception_test_mode::discard ? expect_exception::no : expect_exception::yes, [=] {
            auto rs1 =
                ex::require_started(void_sender{}, ex::require_started_mode::throw_on_unstarted);
            auto rs2 =
                ex::require_started(void_sender{}, ex::require_started_mode::throw_on_unstarted);
            tt::sync_wait(std::move(rs2));    // started
            rs2 = std::move(rs1);             // never started
            discard_if_required(rs2, mode);
        });

    check_exception(
        mode == exception_test_mode::discard ? expect_exception::no : expect_exception::yes, [=] {
            auto rs1 =
                ex::require_started(void_sender{}, ex::require_started_mode::throw_on_unstarted);
            tt::sync_wait(std::move(rs1));    // started
            auto rs2 =
                ex::require_started(void_sender{}, ex::require_started_mode::throw_on_unstarted);
            discard_if_required(rs2, mode);
            rs2 = std::move(rs1);    // never started
        });

    // Copy assignment
    check_exception(
        mode == exception_test_mode::discard ? expect_exception::no : expect_exception::yes, [=] {
            auto rs1 =
                ex::require_started(void_sender{}, ex::require_started_mode::throw_on_unstarted);
            auto rs2 =
                ex::require_started(void_sender{}, ex::require_started_mode::throw_on_unstarted);
            tt::sync_wait(std::move(rs2));    // started
            rs2 = rs1;                        // never started
            tt::sync_wait(std::move(rs1));    // started
            discard_if_required(rs2, mode);
        });

    check_exception(
        mode == exception_test_mode::discard ? expect_exception::no : expect_exception::yes, [=] {
            auto rs1 =
                ex::require_started(void_sender{}, ex::require_started_mode::throw_on_unstarted);
            tt::sync_wait(std::move(rs1));    // started
            auto rs2 = ex::require_started(
                void_sender{}, ex::require_started_mode::throw_on_unstarted);    // never started
            discard_if_required(rs2, mode);
            rs2 = rs1;
        });
#else
    PIKA_UNUSED(mode);
#endif
}

void test_unstarted()
{
#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
    // Connected, but not started
    check_exception(expect_exception::yes, [] {
        auto rs = ex::require_started(void_sender{}, ex::require_started_mode::throw_on_unstarted);
        std::atomic<bool> set_value_called{false};
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(rs), std::move(r));
    });

    check_exception(expect_exception::yes, [] {
        auto rs = ex::require_started(void_sender{}, ex::require_started_mode::throw_on_unstarted);
        std::atomic<bool> set_value_called{false};
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(rs, std::move(r));
        tt::sync_wait(std::move(rs));
    });
#endif
}

void test_any_sender()
{
    check_exception(expect_exception::no,
        [] { tt::sync_wait(ex::unique_any_sender(ex::require_started(void_sender{}))); });

    check_exception(expect_exception::no,
        [] { tt::sync_wait(ex::any_sender(ex::require_started(void_sender{}))); });
}

int main()
{
    test_basics();
    test_pipe_operator();
    test_failure();
    test_adl();
    test_no_exception();
    test_exception(exception_test_mode::no_discard);
    test_exception(exception_test_mode::discard);
    test_unstarted();
    test_any_sender();

    return 0;
}
