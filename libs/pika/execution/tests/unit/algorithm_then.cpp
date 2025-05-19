//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/modules/execution.hpp>
#include <pika/testing.hpp>

#include <pika/execution_base/tests/algorithm_test_utils.hpp>

#include <atomic>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ex = pika::execution::experimental;

struct custom_transformer
{
    std::atomic<bool>& tag_invoke_overload_called;
    std::atomic<bool>& call_operator_called;
    bool throws;

    void operator()() const
    {
        call_operator_called = true;
        if (throws) { throw std::runtime_error("error"); }
    }
};

template <typename S>
auto tag_invoke(ex::then_t, S&& s, custom_transformer t)
{
    t.tag_invoke_overload_called = true;
    return ex::then(std::forward<S>(s), [t = std::move(t)]() { t(); });
}

int main()
{
    // Success path
    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::then(ex::just(), [] {});
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::then(ex::just(0), [](int x) { return ++x; });
        auto f = [](int x) { PIKA_TEST_EQ(x, 1); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::then(ex::just(42), [](auto i) { return i + 1; });    // generic lambda
        auto f = [](int x) { PIKA_TEST_EQ(x, 43); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        int x = 42;
        auto s = ex::then(const_reference_sender<decltype(x)>{x}, [](int i) { return i + 1; });
        auto f = [](int x) { PIKA_TEST_EQ(x, 43); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::then(ex::just(custom_type_non_default_constructible{0}),
            [](custom_type_non_default_constructible x) {
                ++(x.x);
                return x;
            });
        auto f = [](auto x) { PIKA_TEST_EQ(x.x, 1); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::then(ex::just(custom_type_non_default_constructible_non_copyable{0}),
            [](custom_type_non_default_constructible_non_copyable&& x) {
                ++(x.x);
                return std::move(x);
            });
        auto f = [](auto x) { PIKA_TEST_EQ(x.x, 1); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s1 = ex::then(ex::just(0), [](int x) { return ++x; });
        auto s2 = ex::then(std::move(s1), [](int x) { return ++x; });
        auto s3 = ex::then(std::move(s2), [](int x) { return ++x; });
        auto s4 = ex::then(std::move(s3), [](int x) { return ++x; });
        auto f = [](int x) { PIKA_TEST_EQ(x, 4); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s4), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s1 = ex::then(ex::just(), []() { return 3; });
        auto s2 = ex::then(std::move(s1), [](int x) { return x / 1.5; });
        auto s3 = ex::then(std::move(s2), [](double x) { return x / 2; });
        auto s4 = ex::then(std::move(s3), [](int x) { return std::to_string(x); });
        auto f = [](std::string x) { PIKA_TEST_EQ(x, std::string("1")); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s4), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    // This is not recommended, but allowed
    {
        std::atomic<bool> set_value_called{false};
        auto s1 = ex::then(ex::just(), []() -> int& {
            static int x = 3;
            return x;
        });
        auto s2 = ex::then(std::move(s1), [](int& x) -> int& {
            x *= 4;
            return x;
        });
        auto f = [](int& x) { PIKA_TEST_EQ(x, 12); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    // operator| overload
    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::just() | ex::then([]() { return 3; }) |
            ex::then([](int x) { return x / 1.5; }) | ex::then([](double x) { return x / 2; }) |
            ex::then([](int x) { return std::to_string(x); });
        auto f = [](std::string x) { PIKA_TEST_EQ(x, std::string("1")); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), r);
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    // tag_invoke overload
    {
        std::atomic<bool> receiver_set_value_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        std::atomic<bool> custom_transformer_call_operator_called{false};
        auto s = ex::then(ex::just(),
            custom_transformer{
                tag_invoke_overload_called, custom_transformer_call_operator_called, false});
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, receiver_set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(receiver_set_value_called);
        PIKA_TEST(tag_invoke_overload_called);
        PIKA_TEST(custom_transformer_call_operator_called);
    }

    // Failure path
    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::then(ex::just(), [] { throw std::runtime_error("error"); });
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::then(const_reference_error_sender{}, [] {});
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto s1 = ex::then(ex::just(0), [](int x) { return ++x; });
        auto s2 = ex::then(std::move(s1), [](int) -> int { throw std::runtime_error("error"); });
        auto s3 = ex::then(std::move(s2), [](int x) {
            PIKA_TEST(false);
            return ++x;
        });
        auto s4 = ex::then(std::move(s3), [](int x) {
            PIKA_TEST(false);
            return ++x;
        });
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s4), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> receiver_set_error_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        std::atomic<bool> custom_transformer_call_operator_called{false};
        auto s = ex::then(ex::just(),
            custom_transformer{
                tag_invoke_overload_called, custom_transformer_call_operator_called, true});
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, receiver_set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(receiver_set_error_called);
        PIKA_TEST(tag_invoke_overload_called);
        PIKA_TEST(custom_transformer_call_operator_called);
    }

    return 0;
}
