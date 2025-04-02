//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/execution_base/tests/algorithm_test_utils.hpp>
#include <pika/latch.hpp>
#include <pika/modules/execution.hpp>
#include <pika/testing.hpp>

#include <atomic>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ex = pika::execution::experimental;

// This overload is only used to check dispatching. It is not a useful
// implementation.
template <typename Allocator = pika::detail::internal_allocator<>>
auto tag_invoke(ex::ensure_started_t, custom_sender_tag_invoke s, Allocator const& = Allocator{})
{
    s.tag_invoke_overload_called = true;
    return void_sender{};
}

int main()
{
    // Success path
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> started{false};
        auto s1 = ex::then(void_sender{}, [&]() { started = true; });
        auto s2 = ex::ensure_started(std::move(s1));
        PIKA_TEST(started);
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> started{false};
        auto s1 = ex::then(ex::just(0), [&](int x) {
            started = true;
            return x;
        });
        auto s2 = ex::ensure_started(std::move(s1));
        PIKA_TEST(started);
        auto f = [](int x) { PIKA_TEST_EQ(x, 0); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
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
        auto s2 = ex::ensure_started(std::move(s1));
        PIKA_TEST(started);
        auto f = [](auto x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
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
        auto s2 = ex::ensure_started(std::move(s1));
        PIKA_TEST(started);
        auto f = [](auto&& x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        int x = 42;
        auto s1 = const_reference_sender<int>{x};
        auto s2 = ex::ensure_started(std::move(s1));
        auto f = [](auto&& x) { PIKA_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    // operator| overload
    {
        std::atomic<bool> set_value_called{false};
        auto s = void_sender{} | ex::ensure_started();
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    // tag_invoke overload
    {
        std::atomic<bool> receiver_set_value_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        auto s = custom_sender_tag_invoke{tag_invoke_overload_called} | ex::ensure_started();
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
        auto s = error_sender{} | ex::ensure_started();
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto s = const_reference_error_sender{} | ex::ensure_started();
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto s =
            error_sender{} | ex::ensure_started() | ex::ensure_started() | ex::ensure_started();
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    // This test makes sure that a 1. continuation is registered for later invocation in
    // ensure_started and 2. that continuation holds the last reference to the ensure_started shared
    // state. Using a latch to delay the completion of the then operation means that by the time
    // start_detached is connected a continuation has to be registered in ensure_started rather than
    // running the continuation immediately inline. This fulfills 2. Since the ensure_started sender
    // is moved into start_detached the last reference to the ensure_started shared state is
    // released in the continuation of ensure_started.
    //
    // If the released shared state is accessed after it's released this test will fail under
    // valgrind (e.g. while resetting the continuation right after invoking it).
    {
        auto l = std::make_shared<pika::latch>(2);
        auto s = ex::schedule(ex::std_thread_scheduler{}) |
            ex::then([l]() { l->arrive_and_wait(); }) | ex::ensure_started();
        ex::start_detached(std::move(s));
        l->arrive_and_wait();
    }

    // It's allowed to discard the sender from ensure_started
    {
        ex::just() | ex::ensure_started();
    }

    return 0;
}
