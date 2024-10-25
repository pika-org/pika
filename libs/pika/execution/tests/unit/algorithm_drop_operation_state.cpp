//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/modules/execution.hpp>
#include <pika/testing.hpp>

#include <pika/execution_base/tests/algorithm_test_utils.hpp>

#include <atomic>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

int main()
{
    // Success path
    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::drop_operation_state(ex::just());
        constexpr auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::drop_operation_state(ex::just(42));
        constexpr auto f = [](int x) { PIKA_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::drop_operation_state(ex::just(std::string("hello")));
        constexpr auto f = [](std::string x) { PIKA_TEST_EQ(x, std::string("hello")); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::drop_operation_state(ex::just(custom_type_non_default_constructible{42}));
        constexpr auto f = [](auto x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::drop_operation_state(
            ex::just(custom_type_non_default_constructible_non_copyable{42}));
        constexpr auto f = [](auto x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s1 = ex::drop_operation_state(ex::just());
        auto s2 = ex::drop_operation_state(std::move(s1));
        constexpr auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s1 = ex::drop_operation_state(ex::just(32));
        auto s2 = ex::drop_operation_state(std::move(s1));
        constexpr auto f = [](int x) { PIKA_TEST_EQ(x, 32); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        int x = 42;
        auto s = ex::drop_operation_state(const_reference_sender<decltype(x)>{x});
        constexpr auto f = [](int x) { PIKA_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::just(std::tuple<int, std::string>(42, "hello")) | ex::drop_operation_state();
        constexpr auto f = [](std::tuple<int, std::string>&& t) {
            PIKA_TEST_EQ(std::get<int>(t), 42);
            PIKA_TEST_EQ(std::get<std::string>(t), std::string("hello"));
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    // Test that operation state is actually dropped
    {
        auto sp = std::make_shared<int>(42);
        std::weak_ptr<int> sp_weak = sp;

        auto s = ex::just(std::move(sp)) |
            ex::then([&](auto&&) { PIKA_TEST_EQ(sp_weak.use_count(), 1); }) |
            ex::then([&] { PIKA_TEST_EQ(sp_weak.use_count(), 1); }) | ex::drop_operation_state() |
            ex::then([&] { PIKA_TEST_EQ(sp_weak.use_count(), 0); });
        tt::sync_wait(std::move(s));
    }

    // This is a sanity check for the previous case: if we don't use drop_operation_state the
    // shared_ptr should be stay alive even when it's no longer needed
    {
        auto sp = std::make_shared<int>(42);
        std::weak_ptr<int> sp_weak = sp;

        auto s = ex::just(std::move(sp)) |
            ex::then([&](auto&&) { PIKA_TEST_EQ(sp_weak.use_count(), 1); }) |
            ex::then([&]() { PIKA_TEST_EQ(sp_weak.use_count(), 1); });
        tt::sync_wait(std::move(s));
    }

    // operator| overload
    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::just() | ex::drop_operation_state() | ex::drop_operation_state();
        constexpr auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), r);
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    // Failure path
    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::drop_operation_state(
            ex::then(ex::just(), [] { throw std::runtime_error("error"); }));
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::drop_operation_state(const_reference_error_sender{});
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    test_adl_isolation(ex::drop_operation_state(my_namespace::my_sender{}));

    return 0;
}
