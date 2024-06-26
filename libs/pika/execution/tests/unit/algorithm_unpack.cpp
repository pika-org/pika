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
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#if defined(PIKA_HAVE_STDEXEC)
# include <concepts>
#endif

namespace ex = pika::execution::experimental;

namespace test {
    struct my_type
    {
        std::atomic<bool>& tag_invoke_overload_called;
    };

    auto tag_invoke(ex::unpack_t, my_type t)
    {
        t.tag_invoke_overload_called = true;
        return ex::unpack(ex::just(std::tuple(t)));
    }
}    // namespace test

int main()
{
#if defined(PIKA_HAVE_STDEXEC)
    static_assert(std::same_as<ex::value_types_of_t<decltype(ex::unpack(ex::just(std::tuple()))),
                                   ex::empty_env, std::tuple, std::variant>,
        std::variant<std::tuple<>>>);
    static_assert(std::same_as<ex::value_types_of_t<decltype(ex::unpack(ex::just(std::tuple(42)))),
                                   ex::empty_env, std::tuple, std::variant>,
        std::variant<std::tuple<int&&>>>);
    static_assert(
        std::same_as<ex::value_types_of_t<decltype(ex::unpack(std::declval<
                                              const_reference_sender<std::tuple<int>>>())),
                         ex::empty_env, std::tuple, std::variant>,
            std::variant<std::tuple<int const&>>>);
    static_assert(std::same_as<
        ex::value_types_of_t<decltype(ex::unpack(ex::just(std::declval<std::tuple<int&>>()))),
            ex::empty_env, std::tuple, std::variant>,
        std::variant<std::tuple<int&>>>);
    static_assert(
        std::same_as<ex::value_types_of_t<decltype(ex::unpack(ex::just(std::tuple(42, 3.14f)))),
                         ex::empty_env, std::tuple, std::variant>,
            std::variant<std::tuple<int&&, float&&>>>);
#else
    static_assert(
        std::is_same_v<ex::sender_traits<decltype(ex::unpack(ex::just(
                           std::tuple())))>::template value_types<std::tuple, std::variant>,
            std::variant<std::tuple<>>>);
    static_assert(std::is_same_v<ex::sender_traits<decltype(ex::unpack(ex::just(std::tuple(
                                     42))))>::template value_types<std::tuple, std::variant>,
        std::variant<std::tuple<int&&>>>);
    static_assert(std::is_same_v<ex::sender_traits<decltype(ex::unpack(
                                     std::declval<const_reference_sender<std::tuple<int>>>()))>::
                                     template value_types<std::tuple, std::variant>,
        std::variant<std::tuple<int const&>>>);
    static_assert(
        std::is_same_v<ex::sender_traits<decltype(ex::unpack(ex::just(std::declval<
                           std::tuple<int&>>())))>::template value_types<std::tuple, std::variant>,
            std::variant<std::tuple<int&>>>);
    static_assert(std::is_same_v<ex::sender_traits<decltype(ex::unpack(ex::just(std::tuple(
                                     42, 3.14f))))>::template value_types<std::tuple, std::variant>,
        std::variant<std::tuple<int&&, float&&>>>);
#endif

    // Success path
    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::unpack(ex::just(std::tuple()));
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::unpack(ex::just(std::tuple<int>(1)));
        auto f = [](int x) { PIKA_TEST_EQ(x, 1); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::tuple<int> x = 42;
        auto s = ex::unpack(const_reference_sender<decltype(x)>{x});
        auto f = [](int x) { PIKA_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        int x = 42;
        double y = 3.14;
        std::tuple<int&, double&> t{x, y};
        auto s = ex::unpack(const_reference_sender<decltype(t)>{t});
        auto f = [](int& x, double& y) {
            PIKA_TEST_EQ(x, 42);
            PIKA_TEST_EQ(y, 3.14);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        int x = 43;
        std::tuple<int&> t{x};
        auto s = ex::unpack(const_reference_sender<decltype(t)>{t});
        auto f = [](int& x) { PIKA_TEST_EQ(x, 43); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::unpack(ex::just(std::tuple(custom_type_non_default_constructible{42})));
        auto f = [](auto x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::unpack(
            ex::just(std::tuple(custom_type_non_default_constructible_non_copyable{42})));
        auto f = [](auto x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    // operator| overload
    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::just(std::tuple(42, 3.14f)) | ex::unpack() |
            ex::then([](int x, float y) { return std::tuple(x, y); });
        auto f = [](auto t) { PIKA_TEST(t == std::tuple(42, 3.14f)); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), r);
        ex::start(os);
        PIKA_TEST(set_value_called);
    }

    // tag_invoke overload
    {
        std::atomic<bool> receiver_set_value_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        auto s = ex::unpack(test::my_type{tag_invoke_overload_called});
        auto f = [](test::my_type x) { PIKA_TEST(x.tag_invoke_overload_called); };
        auto r = callback_receiver<decltype(f)>{f, receiver_set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(receiver_set_value_called);
        PIKA_TEST(tag_invoke_overload_called);
    }

    // Failure path
    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::unpack(
            ex::then(ex::just(), []() -> std::tuple<> { throw std::runtime_error("error"); }));
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::unpack(ex::then(const_reference_error_sender{}, [] { return std::tuple(); }));
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    test_adl_isolation(ex::unpack(my_namespace::my_sender<std::tuple<my_namespace::my_type>>{}));

    return 0;
}
