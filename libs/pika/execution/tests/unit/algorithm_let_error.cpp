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

// This overload is only used to check dispatching. It is not a useful
// implementation.
template <typename F>
auto tag_invoke(ex::let_error_t, custom_sender_tag_invoke s, F&&)
{
    s.tag_invoke_overload_called = true;
    return void_sender{};
}

int main()
{
    // "Success" path, i.e. let_error gets to handle the error
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> let_error_callback_called{false};
        auto s1 = error_sender{};
        auto s2 = ex::let_error(std::move(s1), [&](std::exception_ptr ep) {
            check_exception_ptr(ep);
            let_error_callback_called = true;
            return void_sender();
        });
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(let_error_callback_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> let_error_callback_called{false};
        auto s1 = const_reference_error_sender{};
        auto s2 = ex::let_error(std::move(s1), [&](std::exception_ptr ep) {
            check_exception_ptr(ep);
            let_error_callback_called = true;
            return void_sender();
        });
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(let_error_callback_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> let_error_callback_called{false};
        auto s1 = error_sender{};
        auto s2 = ex::let_error(std::move(s1), [&](std::exception_ptr ep) {
            check_exception_ptr(ep);
            let_error_callback_called = true;
            return ex::just(42);
        });
        auto f = [](int x) { PIKA_TEST_EQ(x, 42); };
        auto r = callback_receiver<void_callback_helper<decltype(f)>>{
            void_callback_helper<decltype(f)>{f}, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(let_error_callback_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> let_error_callback_called{false};
        auto s1 = error_sender{};
        auto s2 = ex::let_error(std::move(s1), [&](std::exception_ptr ep) {
            check_exception_ptr(ep);
            let_error_callback_called = true;
            return ex::just(custom_type_non_default_constructible{42});
        });
        auto f = [](auto x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<void_callback_helper<decltype(f)>>{
            void_callback_helper<decltype(f)>{f}, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(let_error_callback_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> let_error_callback_called{false};
        auto s1 = error_sender{};
        auto s2 = ex::let_error(std::move(s1), [&](std::exception_ptr ep) {
            check_exception_ptr(ep);
            let_error_callback_called = true;
            return ex::just(custom_type_non_default_constructible_non_copyable{42});
        });
        auto f = [](auto x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<void_callback_helper<decltype(f)>>{
            void_callback_helper<decltype(f)>{f}, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(let_error_callback_called);
    }

    // operator| overload
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> let_error_callback_called{false};
        auto s = error_sender{} | ex::let_error([&](std::exception_ptr ep) {
            check_exception_ptr(ep);
            let_error_callback_called = true;
            return void_sender();
        });
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(let_error_callback_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> let_error_callback_called{false};
        auto s = error_sender{} | ex::let_error([&](std::exception_ptr ep) {
            check_exception_ptr(ep);
            let_error_callback_called = true;
            return ex::just(42);
        });
        auto f = [](int x) { PIKA_TEST_EQ(x, 42); };
        auto r = callback_receiver<void_callback_helper<decltype(f)>>{
            void_callback_helper<decltype(f)>{f}, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(let_error_callback_called);
    }

    // tag_invoke overload
    {
        std::atomic<bool> tag_invoke_overload_called{false};
        custom_sender_tag_invoke{tag_invoke_overload_called} |
            ex::let_error([&](std::exception_ptr) { return ex::just(); });
        PIKA_TEST(tag_invoke_overload_called);
    }

    // "Failure" path, i.e. let_error has no error to handle
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> let_error_callback_called{false};
        auto s1 = ex::just(42);
        auto s2 = ex::let_error(std::move(s1), [&](std::exception_ptr) {
            PIKA_TEST(false);
            return ex::just(43);
        });
        auto f = [](int x) { PIKA_TEST_EQ(x, 42); };
        auto r = callback_receiver<void_callback_helper<decltype(f)>>{
            void_callback_helper<decltype(f)>{f}, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(!let_error_callback_called);
    }
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> let_error_callback_called{false};
        int x = 42;
        auto s1 = const_reference_sender<decltype(x)>{x};
        auto s2 = ex::let_error(std::move(s1), [&](std::exception_ptr) {
            PIKA_TEST(false);
            return ex::just(43);
        });
        auto f = [](int x) { PIKA_TEST_EQ(x, 42); };
        auto r = callback_receiver<void_callback_helper<decltype(f)>>{
            void_callback_helper<decltype(f)>{f}, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(!let_error_callback_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> let_error_callback_called{false};
        auto s1 = ex::just(custom_type_non_default_constructible{42});
        auto s2 = ex::let_error(std::move(s1), [&](std::exception_ptr) {
            PIKA_TEST(false);
            return ex::just(custom_type_non_default_constructible{43});
        });
        auto f = [](auto x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<void_callback_helper<decltype(f)>>{
            void_callback_helper<decltype(f)>{f}, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(!let_error_callback_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> let_error_callback_called{false};
        auto s1 = ex::just(custom_type_non_default_constructible_non_copyable{42});
        auto s2 = ex::let_error(std::move(s1), [&](std::exception_ptr) {
            PIKA_TEST(false);
            return ex::just(custom_type_non_default_constructible_non_copyable{43});
        });
        auto f = [](auto x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<void_callback_helper<decltype(f)>>{
            void_callback_helper<decltype(f)>{f}, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(!let_error_callback_called);
    }

    return 0;
}
