//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/modules/execution.hpp>
#include <pika/testing.hpp>
#include <pika/type_support/unused.hpp>

#include <pika/execution_base/tests/algorithm_test_utils.hpp>

#include <atomic>
#include <exception>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ex = pika::execution::experimental;

// This overload is only used to check dispatching. It is not a useful
// implementation.
template <typename... Ss>
auto tag_invoke(ex::when_all_t, custom_sender_tag_invoke s, Ss&&... ss)
{
    s.tag_invoke_overload_called = true;
    return ex::when_all(std::forward<Ss>(ss)...);
}

int main()
{
    // Success path
    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all(ex::just(42));
        auto f = [](int x) { PIKA_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all(ex::just(42), ex::just(std::string("hello")), ex::just(3.14));
        auto f = [](int x, std::string y, double z) {
            PIKA_TEST_EQ(x, 42);
            PIKA_TEST_EQ(y, std::string("hello"));
            PIKA_TEST_EQ(z, 3.14);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all(ex::just(), ex::just(std::string("hello")), ex::just(3.14));
        auto f = [](std::string y, double z) {
            PIKA_TEST_EQ(y, std::string("hello"));
            PIKA_TEST_EQ(z, 3.14);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all(ex::just(42), ex::just(), ex::just(3.14));
        auto f = [](int x, double z) {
            PIKA_TEST_EQ(x, 42);
            PIKA_TEST_EQ(z, 3.14);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all(ex::just(), ex::just(), ex::just());
        auto f = []() {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all(ex::just(42), ex::just(std::string("hello")), ex::just());
        auto f = [](int x, std::string y) {
            PIKA_TEST_EQ(x, 42);
            PIKA_TEST_EQ(y, std::string("hello"));
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all(ex::just(42, std::string("hello"), 3.14));
        auto f = [](int x, std::string y, double z) {
            PIKA_TEST_EQ(x, 42);
            PIKA_TEST_EQ(y, std::string("hello"));
            PIKA_TEST_EQ(z, 3.14);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all(ex::just(42, std::string("hello")), ex::just(3.14));
        auto f = [](int x, std::string y, double z) {
            PIKA_TEST_EQ(x, 42);
            PIKA_TEST_EQ(y, std::string("hello"));
            PIKA_TEST_EQ(z, 3.14);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all(
            ex::just(), ex::just(42, std::string("hello")), ex::just(), ex::just(3.14), ex::just());
        auto f = [](int x, std::string y, double z) {
            PIKA_TEST_EQ(x, 42);
            PIKA_TEST_EQ(y, std::string("hello"));
            PIKA_TEST_EQ(z, 3.14);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all(ex::just(custom_type_non_default_constructible(42)));
        auto f = [](auto x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all(ex::just(custom_type_non_default_constructible_non_copyable(42)));
        auto f = [](auto x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> receiver_set_value_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        auto s = ex::when_all(custom_sender_tag_invoke{tag_invoke_overload_called}, ex::just(42));
        auto f = [](int x) { PIKA_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, receiver_set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(receiver_set_value_called);
        PIKA_TEST(tag_invoke_overload_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        int x = 42;
        auto s = ex::when_all(const_reference_sender<int>{x});
        auto f = [](auto&& x) { PIKA_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        int x = 42;
        auto s = ex::when_all(const_reference_sender<int>{x});
#if defined(PIKA_HAVE_STDEXEC)
        // The reference implementation does not remove_cvref the types sent.
        PIKA_UNUSED(s);
#else
        using value_types = typename pika::execution::experimental::sender_traits<
            decltype(s)>::template value_types<std::tuple, pika::detail::variant>;
        using expected_value_types = pika::detail::variant<std::tuple<int>>;
        static_assert(std::is_same_v<value_types, expected_value_types>,
            "when_all should remove_cvref the value types sent");
#endif
    }

    // Failure path
    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::when_all(error_sender<double>{});
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::when_all(ex::just(42), error_sender<double>{});
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::when_all(error_sender<double>{}, ex::just(42));
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::when_all(const_reference_error_sender{}, ex::just(42));
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    test_adl_isolation(ex::when_all(my_namespace::my_sender{}));

    return 0;
}
