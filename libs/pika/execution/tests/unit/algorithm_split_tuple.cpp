//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/modules/execution.hpp>
#include <pika/testing.hpp>

#include "algorithm_test_utils.hpp"

#include <atomic>
#include <exception>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

// This overload is only used to check dispatching. It is not a useful
// implementation.
template <typename Allocator = pika::detail::internal_allocator<>>
auto tag_invoke(ex::split_tuple_t, custom_sender_tag_invoke s, Allocator const& = Allocator{})
{
    s.tag_invoke_overload_called = true;
    return void_sender{};
}

int main()
{
    // Success path
    {
        std::atomic<bool> set_value_called{false};
        auto s1 = ex::just(std::tuple(42));
        auto [s2] = ex::split_tuple(std::move(s1));
        auto f = [](int x) { PIKA_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s1 = ex::just(std::tuple(custom_type_non_default_constructible{42}));
        auto [s2] = ex::split_tuple(std::move(s1));
        auto f = [](auto x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s1 = ex::just(std::tuple(custom_type_non_default_constructible_non_copyable{42}));
        auto [s2] = ex::split_tuple(std::move(s1));
        auto f = [](auto x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::tuple<int> x = 42;
        auto s1 = const_reference_sender<std::tuple<int>>{x};
        auto [s2] = ex::split_tuple(std::move(s1));
        auto f = [](auto x) { PIKA_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        auto s1 = ex::just(std::tuple(42, std::string{"hello"}, 3.14));
        auto [s2, s3, s4] = ex::split_tuple(std::move(s1));

        {
            std::atomic<bool> set_value_called{false};
            auto f = [](int x) { PIKA_TEST_EQ(x, 42); };
            auto r = callback_receiver<decltype(f)>{f, set_value_called};
            auto os = ex::connect(std::move(s2), std::move(r));
            ex::start(os);
            PIKA_TEST(set_value_called);
        }

        {
            std::atomic<bool> set_value_called{false};
            auto f = [](std::string x) { PIKA_TEST_EQ(x, std::string{"hello"}); };
            auto r = callback_receiver<decltype(f)>{f, set_value_called};
            auto os = ex::connect(std::move(s3), std::move(r));
            ex::start(os);
            PIKA_TEST(set_value_called);
        }

        {
            std::atomic<bool> set_value_called{false};
            auto f = [](double x) { PIKA_TEST_EQ(x, 3.14); };
            auto r = callback_receiver<decltype(f)>{f, set_value_called};
            auto os = ex::connect(std::move(s4), std::move(r));
            ex::start(os);
            PIKA_TEST(set_value_called);
        }
    }

    // operator| overload
    {
        std::atomic<bool> set_value_called{false};
        auto [s] = ex::just(std::tuple(42)) | ex::split_tuple();
        auto f = [](int x) { PIKA_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    // tag_invoke overload
    {
        std::atomic<bool> receiver_set_value_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        auto s = custom_sender_tag_invoke{tag_invoke_overload_called} | ex::split_tuple();
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, receiver_set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(receiver_set_value_called);
        PIKA_TEST(tag_invoke_overload_called);
    }

    // Failure path
    {
        std::atomic<bool> set_error_called{false};
        auto [s] = error_sender<std::tuple<int>>{} | ex::split_tuple();
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto [s] = ex::split_tuple(const_reference_error_sender<std::tuple<int>>{});
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        auto [s] = ex::split_tuple(my_namespace::my_sender<std::tuple<int>>{});
        test_adl_isolation(s);

        // This is not required by the ADL test, but required by split_tuple. The
        // shared state destructor of the sender returned by split_tuple asserts that
        // the sender has been connected and started before being released.
        tt::sync_wait(std::move(s));
    }

    return 0;
}
