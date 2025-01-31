//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2021 Hartmut Kaiser
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

struct custom_bulk_operation
{
    std::atomic<bool>& tag_invoke_overload_called;
    std::atomic<bool>& call_operator_called;
    std::atomic<int>& call_operator_count;
    bool throws;

    void operator()(int n) const
    {
        PIKA_TEST_EQ(n, call_operator_count.load());

        call_operator_called = true;
        if (n == 3 && throws) { throw std::runtime_error("error"); }
        ++call_operator_count;
    }
};

template <typename S>
auto tag_invoke(ex::bulk_t, S&& s, int num, custom_bulk_operation t)
{
    t.tag_invoke_overload_called = true;
    return ex::bulk(std::forward<S>(s), num, [t = std::move(t)](int n) { t(n); });
}

int main()
{
    // Success path
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<int> set_value_count{0};
        auto s = ex::bulk(ex::just(), 10, [&](int n) {
            PIKA_TEST_EQ(n, set_value_count.load());
            ++set_value_count;
        });
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST_EQ(set_value_count.load(), 10);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<int> set_value_count{0};
        auto s = ex::bulk(ex::just(42), 10, [&](int n, int x) {
            PIKA_TEST_EQ(n, set_value_count.load());
            ++set_value_count;
            return ++x;
        });
        auto f = [](int x) { PIKA_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST_EQ(set_value_count.load(), 10);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<int> set_value_count{0};
        auto s =
            ex::bulk(ex::just(custom_type_non_default_constructible{42}), 10, [&](int n, auto x) {
                PIKA_TEST_EQ(n, set_value_count.load());
                ++set_value_count;
                ++(x.x);
                return x;
            });
        auto f = [](auto&& x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST_EQ(set_value_count.load(), 10);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<int> set_value_count{0};
        auto s = ex::bulk(ex::just(custom_type_non_default_constructible_non_copyable{42}), 10,
            [&](int n, auto&&) {
                PIKA_TEST_EQ(n, set_value_count.load());
                ++set_value_count;
            });
        auto f = [](auto x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST_EQ(set_value_count.load(), 10);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<int> set_value_count{0};
        auto s1 = ex::bulk(ex::just(42), 10, [](int, int x) { return ++x; });
        auto f = [&](int, int) { ++set_value_count; };
        auto s2 = ex::bulk(std::move(s1), 10, f);
        auto s3 = ex::bulk(std::move(s2), 10, f);
        auto s4 = ex::bulk(std::move(s3), 10, f);
        auto f1 = [](int x) { PIKA_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f1)>{f1, set_value_called};
        auto os = ex::connect(std::move(s4), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST_EQ(set_value_count.load(), 30);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<int> set_value_count{0};
        int x = 42;
        auto s = ex::bulk(ex::just(const_reference_sender<decltype(x)>{x}), 10, [&](int n, auto&&) {
            PIKA_TEST_EQ(n, set_value_count.load());
            ++set_value_count;
        });
        auto f = [](auto x) { PIKA_TEST_EQ(x.x.get(), 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST_EQ(set_value_count.load(), 10);
    }

    // operator| overload
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<int> set_value_count{0};
        auto f = [&](int, int) { ++set_value_count; };
        auto s =
            ex::just(42) | ex::bulk(10, f) | ex::bulk(10, f) | ex::bulk(10, f) | ex::bulk(10, f);
        auto f1 = [](int x) { PIKA_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f1)>{f1, set_value_called};
        auto os = ex::connect(std::move(s), r);
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST_EQ(set_value_count.load(), 40);
    }

    // tag_invoke overload
    {
        std::atomic<bool> receiver_set_value_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        std::atomic<bool> custom_bulk_call_operator_called{false};
        std::atomic<int> custom_bulk_call_count{0};
        auto s = ex::bulk(ex::just(), 10,
            custom_bulk_operation{tag_invoke_overload_called, custom_bulk_call_operator_called,
                custom_bulk_call_count, false});
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, receiver_set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(receiver_set_value_called);
        PIKA_TEST(tag_invoke_overload_called);
        PIKA_TEST(custom_bulk_call_operator_called);
        PIKA_TEST_EQ(custom_bulk_call_count.load(), 10);
    }

    // Failure path
    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::bulk(ex::just(), 0, [](int) { throw std::runtime_error("error"); });
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called, true};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(!set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::bulk(ex::just(), 10, [](int n) {
            if (n == 3) throw std::runtime_error("error");
        });
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto s1 = ex::bulk(ex::just(0), 10, [](int, int x) { return ++x; });
        auto s2 = ex::bulk(std::move(s1), 10, [](int n, int x) {
            if (n == 3) throw std::runtime_error("error");
            return x + 1;
        });
        auto s3 = ex::bulk(std::move(s2), 10, [](int, int) { PIKA_TEST(false); });
        auto s4 = ex::bulk(std::move(s3), 10, [](int, int) { PIKA_TEST(false); });
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s4), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> receiver_set_error_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        std::atomic<bool> custom_bulk_call_operator_called{false};
        std::atomic<int> custom_bulk_call_count{0};
        auto s = ex::bulk(ex::just(), 10,
            custom_bulk_operation{tag_invoke_overload_called, custom_bulk_call_operator_called,
                custom_bulk_call_count, true});
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, receiver_set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(receiver_set_error_called);
        PIKA_TEST(tag_invoke_overload_called);
        PIKA_TEST(custom_bulk_call_operator_called);
        PIKA_TEST_EQ(custom_bulk_call_count.load(), 3);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::bulk(const_reference_error_sender{}, 10, [](int) { PIKA_TEST(false); });
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    return 0;
}
