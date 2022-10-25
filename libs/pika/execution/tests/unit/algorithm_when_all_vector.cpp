//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/modules/execution.hpp>
#include <pika/testing.hpp>

#include "algorithm_test_utils.hpp"

#include <atomic>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ex = pika::execution::experimental;

int main()
{
    // Success path
    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all_vector(std::vector<decltype(ex::just())>{});
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        tag_invoke(ex::start, os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all_vector(std::vector<decltype(ex::just(42))>{});
        auto f = [](std::vector<int> v) {
            PIKA_TEST_EQ(v.size(), std::size_t(0));
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        tag_invoke(ex::start, os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all_vector(std::vector{ex::just(42)});
        auto f = [](std::vector<int> v) {
            PIKA_TEST_EQ(v.size(), std::size_t(1));
            PIKA_TEST_EQ(v[0], 42);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        tag_invoke(ex::start, os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        int x = 42;
        auto s =
            ex::when_all_vector(std::vector{const_reference_sender<int>{x}});
        auto f = [](std::vector<int> v) {
            PIKA_TEST_EQ(v.size(), std::size_t(1));
            PIKA_TEST_EQ(v[0], 42);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        tag_invoke(ex::start, os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all_vector(
            std::vector{ex::just(42), ex::just(43), ex::just(44)});
        auto f = [](std::vector<int> v) {
            PIKA_TEST_EQ(v.size(), std::size_t(3));
            PIKA_TEST_EQ(v[0], 42);
            PIKA_TEST_EQ(v[1], 43);
            PIKA_TEST_EQ(v[2], 44);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::vector<ex::unique_any_sender<int>> senders;
        senders.emplace_back(ex::just(42));
        senders.emplace_back(ex::just(43));
        senders.emplace_back(ex::just(44));
        auto s = ex::when_all_vector(std::move(senders));
        auto f = [](std::vector<int> v) {
            PIKA_TEST_EQ(v.size(), std::size_t(3));
            PIKA_TEST_EQ(v[0], 42);
            PIKA_TEST_EQ(v[1], 43);
            PIKA_TEST_EQ(v[2], 44);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::vector<ex::any_sender<double>> senders;
        senders.emplace_back(ex::just(42.0));
        senders.emplace_back(ex::just(43.0));
        senders.emplace_back(ex::just(44.0));
        auto s = ex::when_all_vector(std::move(senders));
        auto f = [](std::vector<double> v) {
            PIKA_TEST_EQ(v.size(), std::size_t(3));
            PIKA_TEST_EQ(v[0], 42.0);
            PIKA_TEST_EQ(v[1], 43.0);
            PIKA_TEST_EQ(v[2], 44.0);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all_vector(std::vector{ex::just()});
        auto f = []() {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        tag_invoke(ex::start, os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all_vector(
            std::vector{ex::just(), ex::just(), ex::just()});
        auto f = []() {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::vector<ex::unique_any_sender<>> senders;
        senders.emplace_back(ex::just());
        senders.emplace_back(ex::just());
        senders.emplace_back(ex::just());
        auto s = ex::when_all_vector(std::move(senders));
        auto f = []() {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::vector<ex::any_sender<>> senders;
        senders.emplace_back(ex::just());
        senders.emplace_back(ex::just());
        senders.emplace_back(ex::just());
        auto s = ex::when_all_vector(std::move(senders));
        auto f = []() {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::vector<ex::any_sender<custom_type_non_default_constructible>>
            senders;
        senders.emplace_back(
            ex::just(custom_type_non_default_constructible{42}));
        senders.emplace_back(
            ex::just(custom_type_non_default_constructible{43}));
        senders.emplace_back(
            ex::just(custom_type_non_default_constructible{44}));
        auto s = ex::when_all_vector(std::move(senders));
        auto f = [](std::vector<custom_type_non_default_constructible> v) {
            PIKA_TEST_EQ(v.size(), std::size_t(3));
            PIKA_TEST_EQ(v[0].x, 42);
            PIKA_TEST_EQ(v[1].x, 43);
            PIKA_TEST_EQ(v[2].x, 44);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::vector<ex::unique_any_sender<
            custom_type_non_default_constructible_non_copyable>>
            senders;
        senders.emplace_back(
            ex::just(custom_type_non_default_constructible_non_copyable{42}));
        senders.emplace_back(
            ex::just(custom_type_non_default_constructible_non_copyable{43}));
        senders.emplace_back(
            ex::just(custom_type_non_default_constructible_non_copyable{44}));
        auto s = ex::when_all_vector(std::move(senders));
        auto f =
            [](std::vector<custom_type_non_default_constructible_non_copyable>
                    v) {
                PIKA_TEST_EQ(v.size(), std::size_t(3));
                PIKA_TEST_EQ(v[0].x, 42);
                PIKA_TEST_EQ(v[1].x, 43);
                PIKA_TEST_EQ(v[2].x, 44);
            };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    // Test a combination with when_all
    {
        std::atomic<bool> set_value_called{false};
        std::vector<decltype(ex::just(std::declval<double>()))> senders1;
        senders1.emplace_back(ex::just(13.0));
        senders1.emplace_back(ex::just(14.0));
        senders1.emplace_back(ex::just(15.0));

        std::vector<ex::any_sender<>> senders2;
        senders2.emplace_back(ex::just());
        senders2.emplace_back(ex::just());

        std::vector<ex::unique_any_sender<int>> senders3;
        senders3.emplace_back(ex::just(42));
        senders3.emplace_back(ex::just(43));
        senders3.emplace_back(ex::just(44));
        senders3.emplace_back(ex::just(45));

        auto s = ex::when_all(ex::when_all_vector(std::move(senders1)),
            ex::when_all_vector(std::move(senders2)),
            ex::when_all_vector(std::move(senders3)));
        auto f = [](std::vector<double> v1, std::vector<int> v3) {
            PIKA_TEST_EQ(v1.size(), std::size_t(3));
            PIKA_TEST_EQ(v1[0], 13.0);
            PIKA_TEST_EQ(v1[1], 14.0);
            PIKA_TEST_EQ(v1[2], 15.0);

            PIKA_TEST_EQ(v3.size(), std::size_t(4));
            PIKA_TEST_EQ(v3[0], 42);
            PIKA_TEST_EQ(v3[1], 43);
            PIKA_TEST_EQ(v3[2], 44);
            PIKA_TEST_EQ(v3[3], 45);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    // Failure path
    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::when_all_vector(std::vector{error_sender<double>{}});
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto s =
            ex::when_all_vector(std::vector{const_reference_error_sender{}});
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        std::vector<ex::unique_any_sender<double>> senders;
        senders.emplace_back(error_sender<double>{});
        senders.emplace_back(ex::just(42.0));
        senders.emplace_back(ex::just(43.0));
        senders.emplace_back(ex::just(44.0));
        auto s = ex::when_all_vector(std::move(senders));
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        std::vector<ex::unique_any_sender<double>> senders;
        senders.emplace_back(error_sender<double>{});
        senders.emplace_back(ex::just(42.0));
        senders.emplace_back(ex::just(43.0));
        senders.emplace_back(ex::just(44.0));
        auto s = ex::when_all_vector(std::move(senders));
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        std::vector<ex::any_sender<double>> senders;
        senders.emplace_back(error_sender<double>{});
        senders.emplace_back(ex::just(42.0));
        senders.emplace_back(ex::just(43.0));
        senders.emplace_back(ex::just(44.0));
        auto s = ex::when_all_vector(std::move(senders));
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        std::vector<ex::any_sender<double>> senders;
        senders.emplace_back(error_sender<double>{});
        senders.emplace_back(ex::just(42.0));
        senders.emplace_back(ex::just(43.0));
        senders.emplace_back(ex::just(44.0));
        auto s = ex::when_all_vector(std::move(senders));
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    test_adl_isolation(
        ex::when_all_vector(std::vector{my_namespace::my_sender{}}));

    return 0;
}
