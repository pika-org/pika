//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/execution_base/any_sender.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/bind_front.hpp>
#include <pika/modules/errors.hpp>
#include <pika/testing.hpp>

#include <pika/execution_base/tests/algorithm_test_utils.hpp>

#include <atomic>
#include <exception>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

struct custom_type_non_copyable
{
    int x;

    custom_type_non_copyable(int x)
      : x(x)
    {
    }
    custom_type_non_copyable() = default;
    custom_type_non_copyable(custom_type_non_copyable&&) = default;
    custom_type_non_copyable& operator=(custom_type_non_copyable&&) = default;
    custom_type_non_copyable(custom_type_non_copyable const&) = delete;
    custom_type_non_copyable& operator=(custom_type_non_copyable const&) = delete;
};

template <typename... Ts>
struct non_copyable_sender
{
    PIKA_STDEXEC_SENDER_CONCEPT

    std::tuple<std::decay_t<Ts>...> ts;

    template <template <class...> class Tuple, template <class...> class Variant>
    using value_types = Variant<Tuple<Ts...>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    using completion_signatures = pika::execution::experimental::completion_signatures<
        pika::execution::experimental::set_value_t(Ts...),
        pika::execution::experimental::set_error_t(std::exception_ptr)>;

    non_copyable_sender() = default;
    template <typename T,
        typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, non_copyable_sender>>>
    non_copyable_sender(T&& t)
      : ts(std::forward<T>(t))
    {
    }
    template <typename T1, typename T2, typename... Ts_>
    non_copyable_sender(T1&& t1, T2&& t2, Ts_&&... ts)
      : ts(std::forward<T1>(t1), std::forward<T2>(t2), std::forward<Ts_>(ts)...)
    {
    }
    non_copyable_sender(non_copyable_sender&&) = default;
    non_copyable_sender(non_copyable_sender const&) = delete;
    non_copyable_sender& operator=(non_copyable_sender&&) = default;
    non_copyable_sender& operator=(non_copyable_sender const&) = delete;

    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;
        std::tuple<std::decay_t<Ts>...> ts;

        friend void tag_invoke(pika::execution::experimental::start_t, operation_state& os) noexcept
        {
            std::apply(pika::util::detail::bind_front(
                           pika::execution::experimental::set_value, std::move(os.r)),
                std::move(os.ts));
        };
    };

    template <typename R>
    friend operation_state<R>
    tag_invoke(pika::execution::experimental::connect_t, non_copyable_sender&& s, R&& r) noexcept
    {
        return {std::forward<R>(r), std::move(s.ts)};
    }
};

template <typename... Ts>
struct sender
{
    PIKA_STDEXEC_SENDER_CONCEPT

    std::tuple<std::decay_t<Ts>...> ts;

    template <template <class...> class Tuple, template <class...> class Variant>
    using value_types = Variant<Tuple<Ts...>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    using completion_signatures = pika::execution::experimental::completion_signatures<
        pika::execution::experimental::set_value_t(Ts...),
        pika::execution::experimental::set_error_t(std::exception_ptr)>;

    sender() = default;
    template <typename T, typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, sender>>>
    sender(T&& t)
      : ts(std::forward<T>(t))
    {
    }
    template <typename T1, typename T2, typename... Ts_>
    sender(T1&& t1, T2&& t2, Ts_&&... ts)
      : ts(std::forward<T1>(t1), std::forward<T2>(t2), std::forward<Ts_>(ts)...)
    {
    }
    sender(sender&&) = default;
    sender(sender const&) = default;
    sender& operator=(sender&&) = default;
    sender& operator=(sender const&) = default;

    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;
        std::tuple<std::decay_t<Ts>...> ts;

        friend void tag_invoke(pika::execution::experimental::start_t, operation_state& os) noexcept
        {
            std::apply(pika::util::detail::bind_front(
                           pika::execution::experimental::set_value, std::move(os.r)),
                std::move(os.ts));
        };
    };

    template <typename R>
    friend operation_state<R>
    tag_invoke(pika::execution::experimental::connect_t, sender&& s, R&& r)
    {
        return {std::forward<R>(r), std::move(s.ts)};
    }

    template <typename R>
    friend operation_state<R>
    tag_invoke(pika::execution::experimental::connect_t, sender const& s, R&& r)
    {
        return {std::forward<R>(r), s.ts};
    }
};

template <typename... Ts>
struct large_non_copyable_sender : non_copyable_sender<Ts...>
{
    // This padding only needs to be larger than the embedded storage in
    // any_sender. Adjust if needed.
    unsigned char padding[128] = {0};

    large_non_copyable_sender() = default;
    template <typename T,
        typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, large_non_copyable_sender>>>
    large_non_copyable_sender(T&& t)
      : non_copyable_sender<Ts...>(std::forward<T>(t))
    {
    }
    template <typename T1, typename T2, typename... Ts_>
    large_non_copyable_sender(T1&& t1, T2&& t2, Ts_&&... ts)
      : non_copyable_sender<Ts...>(
            std::forward<T1>(t1), std::forward<T2>(t2), std::forward<Ts_>(ts)...)
    {
    }
    large_non_copyable_sender(large_non_copyable_sender&&) = default;
    large_non_copyable_sender(large_non_copyable_sender const&) = delete;
    large_non_copyable_sender& operator=(large_non_copyable_sender&&) = default;
    large_non_copyable_sender& operator=(large_non_copyable_sender const&) = delete;
};

template <typename... Ts>
struct large_sender : sender<Ts...>
{
    // This padding only needs to be larger than the embedded storage in
    // any_sender. Adjust if needed.
    unsigned char padding[128] = {0};

    large_sender() = default;
    template <typename T,
        typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, large_sender>>>
    large_sender(T&& t)
      : sender<Ts...>(std::forward<T>(t))
    {
    }
    template <typename T1, typename T2, typename... Ts_>
    large_sender(T1&& t1, T2&& t2, Ts_&&... ts)
      : sender<Ts...>(std::forward<T1>(t1), std::forward<T2>(t2), std::forward<Ts_>(ts)...)
    {
    }
    large_sender(large_sender&&) = default;
    large_sender(large_sender const&) = default;
    large_sender& operator=(large_sender&&) = default;
    large_sender& operator=(large_sender const&) = default;
};

struct error_receiver
{
    PIKA_STDEXEC_RECEIVER_CONCEPT

    std::atomic<bool>& set_error_called;

    friend void tag_invoke(pika::execution::experimental::set_error_t, error_receiver&& r,
        std::exception_ptr&& e) noexcept
    {
        try
        {
            std::rethrow_exception(std::move(e));
        }
        catch (std::runtime_error const& e)
        {
            PIKA_TEST_EQ(std::string(e.what()), std::string("error"));
        }
        catch (...)
        {
            PIKA_TEST(false);
        }
        r.set_error_called = true;
    }

    friend void tag_invoke(pika::execution::experimental::set_stopped_t, error_receiver&&) noexcept
    {
        PIKA_TEST(false);
    };

    template <typename... Ts>
    void set_value(Ts&&...) && noexcept
    {
        PIKA_TEST(false);
    }

    friend constexpr pika::execution::experimental::empty_env tag_invoke(
        pika::execution::experimental::get_env_t, error_receiver const&) noexcept
    {
        return {};
    }
};

template <template <typename...> typename Sender, typename... Ts, typename F>
void test_any_sender(F&& f, Ts&&... ts)
{
    static_assert(std::is_copy_constructible_v<Sender<Ts...>>,
        "This test requires the sender to be copy constructible.");

    Sender<std::decay_t<Ts>...> s{std::forward<Ts>(ts)...};

    ex::any_sender<std::decay_t<Ts>...> as1{s};
    auto as2 = as1;

    PIKA_TEST(!as1.empty());
    PIKA_TEST(!as2.empty());
    PIKA_TEST(as1);
    PIKA_TEST(as2);

    // We should be able to connect both as1 and as2 multiple times; set_value
    // should always be called
    {
        std::atomic<bool> set_value_called{false};
        auto os = ex::connect(as1, callback_receiver<F>{f, set_value_called});
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(!as1.empty());
        PIKA_TEST(as1);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto os = ex::connect(std::move(as1), callback_receiver<F>{f, set_value_called});
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(as1.empty());
        PIKA_TEST(!as1);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto os = ex::connect(as2, callback_receiver<F>{f, set_value_called});
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(!as2.empty());
        PIKA_TEST(as2);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto os = ex::connect(std::move(as2), callback_receiver<F>{f, set_value_called});
        ex::start(os);
        PIKA_TEST(set_value_called);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        PIKA_TEST(as2.empty());
        PIKA_TEST(!as2);
    }

    // as1 and as2 have been moved so we always expect an exception here
    {
        std::atomic<bool> set_value_called{false};
        try
        {
            auto os = ex::connect(
                // NOLINTNEXTLINE(bugprone-use-after-move)
                std::move(as1), callback_receiver<F>{f, set_value_called});
            PIKA_TEST(false);
            ex::start(os);
        }
        catch (pika::exception const& e)
        {
            PIKA_TEST_EQ(e.get_error(), pika::error::bad_function_call);
        }
        catch (...)
        {
            PIKA_TEST(false);
        }
        PIKA_TEST(!set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        try
        {
            auto os = ex::connect(
                // NOLINTNEXTLINE(bugprone-use-after-move)
                std::move(as2), callback_receiver<F>{f, set_value_called});
            PIKA_TEST(false);
            ex::start(os);
        }
        catch (pika::exception const& e)
        {
            PIKA_TEST_EQ(e.get_error(), pika::error::bad_function_call);
        }
        catch (...)
        {
            PIKA_TEST(false);
        }
        PIKA_TEST(!set_value_called);
    }
}

template <template <typename...> typename Sender, typename... Ts, typename F>
void test_unique_any_sender(F&& f, Ts&&... ts)
{
    Sender<std::decay_t<Ts>...> s{std::forward<Ts>(ts)...};

    ex::unique_any_sender<std::decay_t<Ts>...> as1{std::move(s)};
    auto as2 = std::move(as1);

    PIKA_TEST(as1.empty());
    PIKA_TEST(!as2.empty());
    PIKA_TEST(!as1);
    PIKA_TEST(as2);

    // We expect set_value to be called here
    {
        std::atomic<bool> set_value_called = false;
        auto os = ex::connect(std::move(as2), callback_receiver<F>{f, set_value_called});
        ex::start(os);
        PIKA_TEST(set_value_called);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        PIKA_TEST(as2.empty());
        PIKA_TEST(!as2);
    }

    // as1 has been moved so we always expect an exception here
    {
        std::atomic<bool> set_value_called{false};
        try
        {
            auto os = ex::connect(
                // NOLINTNEXTLINE(bugprone-use-after-move)
                std::move(as1), callback_receiver<F>{f, set_value_called});
            PIKA_TEST(false);
            ex::start(os);
        }
        catch (pika::exception const& e)
        {
            PIKA_TEST_EQ(e.get_error(), pika::error::bad_function_call);
        }
        catch (...)
        {
            PIKA_TEST(false);
        }
        PIKA_TEST(!set_value_called);
    }
}

void test_any_sender_set_error()
{
    error_sender<> s;

    ex::any_sender<> as1{std::move(s)};
    auto as2 = as1;

    // We should be able to connect the sender multiple times; set_error should
    // always be called
    {
        std::atomic<bool> set_error_called{false};
        auto os = ex::connect(as1, error_receiver{set_error_called});
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto os = ex::connect(std::move(as1), error_receiver{set_error_called});
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto os = ex::connect(as2, error_receiver{set_error_called});
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto os = ex::connect(std::move(as2), error_receiver{set_error_called});
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    // as1 and as2 have been moved so we always expect an exception here
    {
        std::atomic<bool> set_error_called{false};
        try
        {
            auto os =
                // NOLINTNEXTLINE(bugprone-use-after-move)
                ex::connect(std::move(as1), error_receiver{set_error_called});
            PIKA_TEST(false);
            ex::start(os);
        }
        catch (pika::exception const& e)
        {
            PIKA_TEST_EQ(e.get_error(), pika::error::bad_function_call);
        }
        catch (...)
        {
            PIKA_TEST(false);
        }
        PIKA_TEST(!set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        try
        {
            auto os =
                // NOLINTNEXTLINE(bugprone-use-after-move)
                ex::connect(std::move(as2), error_receiver{set_error_called});
            PIKA_TEST(false);
            ex::start(os);
        }
        catch (pika::exception const& e)
        {
            PIKA_TEST_EQ(e.get_error(), pika::error::bad_function_call);
        }
        catch (...)
        {
            PIKA_TEST(false);
        }
        PIKA_TEST(!set_error_called);
    }
}

void test_unique_any_sender_set_error()
{
    error_sender<> s;

    ex::unique_any_sender<> as1{std::move(s)};
    auto as2 = std::move(as1);

    // We expect set_error to be called here
    {
        std::atomic<bool> set_error_called{false};
        auto os = ex::connect(std::move(as2), error_receiver{set_error_called});
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    // as1 has been moved so we always expect an exception here
    {
        std::atomic<bool> set_error_called{false};
        try
        {
            auto os =
                // NOLINTNEXTLINE(bugprone-use-after-move)
                ex::connect(std::move(as1), error_receiver{set_error_called});
            PIKA_TEST(false);
            ex::start(os);
        }
        catch (pika::exception const& e)
        {
            PIKA_TEST_EQ(e.get_error(), pika::error::bad_function_call);
        }
        catch (...)
        {
            PIKA_TEST(false);
        }
        PIKA_TEST(!set_error_called);
    }
}

// (unique_)any_sender instantiates a set_stopped call, and should also
// advertise that it can complete with set_stopped. transfer from stdexec is one
// algorithm that requires that the advertised signatures match what is actually
// instantiated so we use that as a test here. This will fail to compile if
// (unique_)any_sender doesn't have set_stopped_t in its completion signatures.
void test_any_sender_set_stopped()
{
    ex::any_sender<> as{ex::just()};
    tt::sync_wait(ex::continues_on(std::move(as), ex::std_thread_scheduler{}));
}

void test_unique_any_sender_set_stopped()
{
    ex::any_sender<> as{ex::just()};
    tt::sync_wait(ex::continues_on(std::move(as), ex::std_thread_scheduler{}));
}

// This tests that the empty vtable types used in the implementation of any_*
// are not destroyed too early. We use ensure_started inside the function to
// trigger the use of the empty vtables for any_receiver and
// any_operation_state. If the empty vtables are function-local statics they
// would get constructed after global_*any_sender is constructed, and thus
// destroyed before global_*any_sender is destroyed. This will typically lead to
// a segfault. If the empty vtables are (constant) global variables they should
// be constructed before global_*any_sender is constructed and destroyed after
// global_*any_sender is destroyed.
ex::unique_any_sender<> global_unique_any_sender{ex::just()};
ex::any_sender<> global_any_sender{ex::just()};

// This helper only makes sure that the sender returned from split is actually
// started before being destructed.
struct wait_globals
{
    ~wait_globals() { tt::sync_wait(std::move(global_any_sender)); }
} waiter{};

void test_globals()
{
    global_unique_any_sender = std::move(global_unique_any_sender) | ex::ensure_started();
    global_any_sender = std::move(global_any_sender) | ex::ensure_started() | ex::split();
}

void test_empty_any_sender()
{
    {
        ex::unique_any_sender<> uas{};
        ex::any_sender<> as{};

        PIKA_TEST(uas.empty());
        PIKA_TEST(!uas);
        PIKA_TEST(as.empty());
        PIKA_TEST(!as);
    }

    {
        ex::unique_any_sender<> uas{ex::just()};
        ex::any_sender<> as{ex::just()};

        PIKA_TEST(!uas.empty());
        PIKA_TEST(uas);
        PIKA_TEST(!as.empty());
        PIKA_TEST(as);

        uas.reset(ex::just());
        as.reset(ex::just());

        PIKA_TEST(!uas.empty());
        PIKA_TEST(uas);
        PIKA_TEST(!as.empty());
        PIKA_TEST(as);

        uas.reset();
        as.reset();

        PIKA_TEST(uas.empty());
        PIKA_TEST(!uas);
        PIKA_TEST(as.empty());
        PIKA_TEST(!as);

        uas.reset(ex::unique_any_sender<>{});
        as.reset(ex::any_sender<>{});

        PIKA_TEST(uas.empty());
        PIKA_TEST(!uas);
        PIKA_TEST(as.empty());
        PIKA_TEST(!as);
    }
}

void test_make_any_sender()
{
    static_assert(
        std::is_same_v<decltype(ex::make_unique_any_sender(ex::just())), ex::unique_any_sender<>>);
    static_assert(std::is_same_v<decltype(ex::make_any_sender(ex::just())), ex::any_sender<>>);
    static_assert(std::is_same_v<decltype(ex::make_unique_any_sender(ex::just(3))),
        ex::unique_any_sender<int>>);
    static_assert(std::is_same_v<decltype(ex::make_any_sender(ex::just(42))), ex::any_sender<int>>);
    static_assert(
        std::is_same_v<decltype(ex::make_unique_any_sender(ex::just(3, std::string("hello")))),
            ex::unique_any_sender<int, std::string>>);
    static_assert(std::is_same_v<decltype(ex::make_any_sender(ex::just(42, std::string("bye")))),
        ex::any_sender<int, std::string>>);
    static_assert(!std::is_same_v<decltype(ex::make_any_sender(ex::just(42, std::string("bye")))),
                  ex::any_sender<int, std::string, double>>);
}

void test_when_all()
{
    ex::any_sender<> as1{ex::just()};
    ex::any_sender<int> as2{ex::just(42)};
    ex::any_sender<int> as3{ex::when_all(std::move(as1), std::move(as2))};

    tt::sync_wait(std::move(as3));
}

// To control when the move constructor will throw, exception is caught in the set_value of
// any_receiver
bool throwing = false;

struct throwing_constructor
{
    throwing_constructor() noexcept(false) {}
    throwing_constructor(throwing_constructor&&) noexcept(false)
    {
        if (throwing) throw std::runtime_error("error");
    };
    // to allow use with any_sender
    throwing_constructor(throwing_constructor const&) = default;
    throwing_constructor(throwing_constructor&) noexcept(false)
    {
        if (throwing) throw std::runtime_error("error");
    };
};

void test_throwing_constructor()
{
    {
        // Throwing move with unique_any_sender
        throwing = false;
        std::atomic<bool> set_error_called{false};
        ex::unique_any_sender<throwing_constructor> as1 = ex::just(throwing_constructor());
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(as1), std::move(r));
        throwing = true;
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        // Throwing move with any_sender
        throwing = false;
        std::atomic<bool> set_error_called{false};
        ex::any_sender<throwing_constructor> as1 = ex::just(throwing_constructor());
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(as1), std::move(r));
        throwing = true;
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        // Throwing copy with unique_any_sender
        throwing = false;
        auto throwing_object = throwing_constructor();
        auto s = const_reference_sender<throwing_constructor>{throwing_object};
        std::atomic<bool> set_error_called{false};
        ex::unique_any_sender<throwing_constructor> as1 = s;
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(as1), std::move(r));
        throwing = true;
        ex::start(os);
        PIKA_TEST(set_error_called);
    }

    {
        // Throwing copy with any_sender
        throwing = false;
        auto throwing_object = throwing_constructor();
        auto s = const_reference_sender<throwing_constructor>{throwing_object};
        std::atomic<bool> set_error_called{false};
        ex::any_sender<throwing_constructor> as1 = s;
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(as1), std::move(r));
        throwing = true;
        ex::start(os);
        PIKA_TEST(set_error_called);
    }
}

void test_const_reference()
{
    int x = 42;
    [[maybe_unused]] ex::unique_any_sender<int> as1 = const_reference_sender<int>{x};
    [[maybe_unused]] ex::any_sender<int> as2 = const_reference_sender<int>{x};
    PIKA_TEST(true);
}

void test_let_value()
{
    {
        ex::unique_any_sender<> as = ex::just();
        auto s = ex::let_value(std::move(as), []() { return ex::just(); });
        tt::sync_wait(std::move(s));
    }

    {
        auto s = ex::let_value(ex::just(), []() -> ex::unique_any_sender<> { return ex::just(); });
        tt::sync_wait(std::move(s));
    }

    {
        ex::unique_any_sender<> s = ex::let_value(ex::just(), []() { return ex::just(); });
        tt::sync_wait(std::move(s));
    }

    {
        ex::unique_any_sender<> as = ex::just();
        ex::unique_any_sender<> s =
            ex::let_value(std::move(as), []() -> ex::unique_any_sender<> { return ex::just(); });
        tt::sync_wait(std::move(s));
    }

    {
        ex::unique_any_sender<> s = ex::let_value(ex::make_unique_any_sender(ex::just()),
            [s = ex::make_unique_any_sender(ex::just())]() mutable {
                return ex::make_unique_any_sender(ex::let_value(
                    std::move(s), []() { return ex::make_unique_any_sender(ex::just()); }));
            });
        tt::sync_wait(std::move(s));
    }
}

template <template <typename...> typename Sender, typename... Ts, typename F>
void test_any_sender_to_unique_any_sender(F&& f, Ts&&... ts)
{
    static_assert(std::is_copy_constructible_v<Sender<Ts...>>,
        "This test requires the sender to be copy constructible.");

    Sender<std::decay_t<Ts>...> s{std::forward<Ts>(ts)...};

    ex::any_sender<std::decay_t<Ts>...> as1{s};
    ex::any_sender<std::decay_t<Ts>...> as2{s};

    ex::unique_any_sender<std::decay_t<Ts>...> as3{as1};
    ex::unique_any_sender<std::decay_t<Ts>...> as4;
    as4 = as2;

    ex::unique_any_sender<std::decay_t<Ts>...> as5{std::move(as1)};
    ex::unique_any_sender<std::decay_t<Ts>...> as6;
    as6 = std::move(as2);

    PIKA_TEST(as1.empty());
    PIKA_TEST(as2.empty());
    PIKA_TEST(as3);
    PIKA_TEST(as4);
    PIKA_TEST(as5);
    PIKA_TEST(as6);

    // We expect set_value to be called here for as3 to as6
    auto check_set_value = [&](auto&& s) {
        std::atomic<bool> set_value_called = false;
        auto os =
            ex::connect(std::forward<decltype(s)>(s), callback_receiver<F>{f, set_value_called});
        ex::start(os);
        PIKA_TEST(set_value_called);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        PIKA_TEST(s.empty());
        PIKA_TEST(!s);
    };

    check_set_value(std::move(as3));
    check_set_value(std::move(as4));
    check_set_value(std::move(as5));
    check_set_value(std::move(as6));

    // All senders have been moved now so we expect exceptions here
    auto check_exception = [&](auto&& s) {
        std::atomic<bool> set_value_called{false};
        try
        {
            auto os = ex::connect(
                // NOLINTNEXTLINE(bugprone-use-after-move)
                std::forward<decltype(s)>(s), callback_receiver<F>{f, set_value_called});
            PIKA_TEST(false);
            ex::start(os);
        }
        catch (pika::exception const& e)
        {
            PIKA_TEST_EQ(e.get_error(), pika::error::bad_function_call);
        }
        catch (...)
        {
            PIKA_TEST(false);
        }
        PIKA_TEST(!set_value_called);
    };

    // NOLINTNEXTLINE(bugprone-use-after-move)
    check_exception(std::move(as1));
    // NOLINTNEXTLINE(bugprone-use-after-move)
    check_exception(std::move(as2));
    // NOLINTNEXTLINE(bugprone-use-after-move)
    check_exception(std::move(as3));
    // NOLINTNEXTLINE(bugprone-use-after-move)
    check_exception(std::move(as4));
    // NOLINTNEXTLINE(bugprone-use-after-move)
    check_exception(std::move(as5));
    // NOLINTNEXTLINE(bugprone-use-after-move)
    check_exception(std::move(as6));
}

int main()
{
    // We can only wrap copyable senders in any_sender
    test_any_sender<sender>([] {});
    test_any_sender<sender, int>([](int x) { PIKA_TEST_EQ(x, 42); }, 42);
    test_any_sender<sender, int, double>(
        [](int x, double y) {
            PIKA_TEST_EQ(x, 42);
            PIKA_TEST_EQ(y, 3.14);
        },
        42, 3.14);

    test_any_sender<large_sender>([] {});
    test_any_sender<large_sender, int>([](int x) { PIKA_TEST_EQ(x, 42); }, 42);
    test_any_sender<large_sender, int, double>(
        [](int x, double y) {
            PIKA_TEST_EQ(x, 42);
            PIKA_TEST_EQ(y, 3.14);
        },
        42, 3.14);

    // We can wrap both copyable and non-copyable senders in unique_any_sender
    test_unique_any_sender<sender>([] {});
    test_unique_any_sender<sender, int>([](int x) { PIKA_TEST_EQ(x, 42); }, 42);
    test_unique_any_sender<sender, int, double>(
        [](int x, double y) {
            PIKA_TEST_EQ(x, 42);
            PIKA_TEST_EQ(y, 3.14);
        },
        42, 3.14);

    test_unique_any_sender<large_sender>([] {});
    test_unique_any_sender<large_sender, int>([](int x) { PIKA_TEST_EQ(x, 42); }, 42);
    test_unique_any_sender<large_sender, int, double>(
        [](int x, double y) {
            PIKA_TEST_EQ(x, 42);
            PIKA_TEST_EQ(y, 3.14);
        },
        42, 3.14);

    test_unique_any_sender<non_copyable_sender>([] {});
    test_unique_any_sender<non_copyable_sender, int>([](int x) { PIKA_TEST_EQ(x, 42); }, 42);
    test_unique_any_sender<non_copyable_sender, int, double>(
        [](int x, double y) {
            PIKA_TEST_EQ(x, 42);
            PIKA_TEST_EQ(y, 3.14);
        },
        42, 3.14);
    test_unique_any_sender<non_copyable_sender, int, double, custom_type_non_copyable>(
        [](int x, double y, custom_type_non_copyable z) {
            PIKA_TEST_EQ(x, 42);
            PIKA_TEST_EQ(y, 3.14);
            PIKA_TEST_EQ(z.x, 43);
        },
        42, 3.14, custom_type_non_copyable(43));

    test_unique_any_sender<large_non_copyable_sender>([] {});
    test_unique_any_sender<large_non_copyable_sender, int>([](int x) { PIKA_TEST_EQ(x, 42); }, 42);
    test_unique_any_sender<large_non_copyable_sender, int, double>(
        [](int x, double y) {
            PIKA_TEST_EQ(x, 42);
            PIKA_TEST_EQ(y, 3.14);
        },
        42, 3.14);
    test_unique_any_sender<large_non_copyable_sender, int, double, custom_type_non_copyable>(
        [](int x, double y, custom_type_non_copyable z) {
            PIKA_TEST_EQ(x, 42);
            PIKA_TEST_EQ(y, 3.14);
            PIKA_TEST_EQ(z.x, 43);
        },
        42, 3.14, custom_type_non_copyable(43));

    // Failure paths
    test_any_sender_set_error();
    test_unique_any_sender_set_error();

    // set_stopped
    test_any_sender_set_stopped();
    test_unique_any_sender_set_stopped();

    // Test use of *any_* in globals
    test_globals();

    // Test that default constructed senders are empty
    test_empty_any_sender();

    // Test deducing value types with make(_unique)_any_sender
    test_make_any_sender();

    // Test using any_senders together with when_all
    test_when_all();

    // Test using {unique_,}any_sender with a just sender of move object that can throw
    test_throwing_constructor();

    // Test using {unique_,}any_sender with a just sender of a const reference
    test_const_reference();

    // Test interaction with let_value
    test_let_value();

    // Test construction of unique_any_sender from any_sender
    test_any_sender_to_unique_any_sender<sender>([] {});
    test_any_sender_to_unique_any_sender<sender, int>([](int x) { PIKA_TEST_EQ(x, 42); }, 42);
    test_any_sender_to_unique_any_sender<sender, int, double>(
        [](int x, double y) {
            PIKA_TEST_EQ(x, 42);
            PIKA_TEST_EQ(y, 3.14);
        },
        42, 3.14);

    test_any_sender_to_unique_any_sender<large_sender>([] {});
    test_any_sender_to_unique_any_sender<large_sender, int>([](int x) { PIKA_TEST_EQ(x, 42); }, 42);
    test_any_sender_to_unique_any_sender<large_sender, int, double>(
        [](int x, double y) {
            PIKA_TEST_EQ(x, 42);
            PIKA_TEST_EQ(y, 3.14);
        },
        42, 3.14);

    return 0;
}
