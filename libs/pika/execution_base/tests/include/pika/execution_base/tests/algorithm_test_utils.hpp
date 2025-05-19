//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/functional/tag_invoke.hpp>
#include <pika/modules/execution.hpp>
#include <pika/testing.hpp>

#include <atomic>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#pragma once

struct void_sender
{
    PIKA_STDEXEC_SENDER_CONCEPT

    template <template <typename...> class Tuple, template <typename...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <typename...> class Variant>
    using error_types = Variant<>;

    static constexpr bool sends_done = false;

    using completion_signatures = pika::execution::experimental::completion_signatures<
        pika::execution::experimental::set_value_t()>;

    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;
        void start() & noexcept { pika::execution::experimental::set_value(std::move(r)); }
    };

    template <typename R>
    operation_state<R> connect(R&& r) const
    {
        return {std::forward<R>(r)};
    }
};

template <typename... Ts>
struct error_sender
{
    PIKA_STDEXEC_SENDER_CONCEPT

    template <template <typename...> class Tuple, template <typename...> class Variant>
    using value_types = Variant<Tuple<Ts...>>;

    template <template <typename...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    using completion_signatures = pika::execution::experimental::completion_signatures<
        pika::execution::experimental::set_value_t(Ts...),
        pika::execution::experimental::set_error_t(std::exception_ptr)>;

    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;
        void start() & noexcept
        {
            try
            {
                throw std::runtime_error("error");
            }
            catch (...)
            {
                pika::execution::experimental::set_error(std::move(r), std::current_exception());
            }
        }
    };

    template <typename R>
    operation_state<R> connect(R&& r) const
    {
        return {std::forward<R>(r)};
    }
};

template <typename... Ts>
struct const_reference_error_sender
{
    PIKA_STDEXEC_SENDER_CONCEPT

    template <template <class...> class Tuple, template <class...> class Variant>
    using value_types = Variant<Tuple<Ts...>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr const&>;

    static constexpr bool sends_done = false;

    using completion_signatures = pika::execution::experimental::completion_signatures<
        pika::execution::experimental::set_value_t(Ts...),
        pika::execution::experimental::set_error_t(std::exception_ptr const&)>;

    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;
        void start() & noexcept
        {
            auto const e = std::make_exception_ptr(std::runtime_error("error"));
            pika::execution::experimental::set_error(std::move(r), e);
        }
    };

    template <typename R>
    operation_state<R> connect(R&& r) const
    {
        return {std::forward<R>(r)};
    }
};

template <typename F>
struct callback_receiver
{
    PIKA_STDEXEC_RECEIVER_CONCEPT

    std::decay_t<F> f;
    std::atomic<bool>& set_value_called;

    template <typename E>
    void set_error(E&&) && noexcept
    {
        PIKA_TEST(false);
    }

    friend void tag_invoke(
        pika::execution::experimental::set_stopped_t, callback_receiver&&) noexcept
    {
        PIKA_TEST(false);
    };

    template <typename... Ts>
    auto set_value(Ts&&... ts) && noexcept
        -> decltype(PIKA_INVOKE(std::declval<std::decay_t<F>>(), std::forward<Ts>(ts)...), void())
    {
        auto r = std::move(*this);
        PIKA_INVOKE(r.f, std::forward<Ts>(ts)...);
        r.set_value_called = true;
    }

    constexpr pika::execution::experimental::empty_env get_env() const& noexcept { return {}; }
};

template <typename F>
struct error_callback_receiver
{
    PIKA_STDEXEC_RECEIVER_CONCEPT

    std::decay_t<F> f;
    std::atomic<bool>& set_error_called;
    bool expect_set_value = false;

    template <typename E>
    void set_error(E&& e) && noexcept
    {
        auto r = std::move(*this);
        PIKA_INVOKE(r.f, std::forward<E>(e));
        r.set_error_called = true;
    }

    friend void tag_invoke(
        pika::execution::experimental::set_stopped_t, error_callback_receiver&&) noexcept
    {
        PIKA_TEST(false);
    };

    template <typename... Ts>
    void set_value(Ts&&...) && noexcept
    {
        auto r = std::move(*this);
        PIKA_TEST(r.expect_set_value);
    }

    constexpr pika::execution::experimental::empty_env get_env() const& noexcept { return {}; }
};

template <typename F>
struct void_callback_helper
{
    std::decay_t<F> f;

    // This overload is only used to satisfy tests that have a predecessor that
    // can send void, but never does in practice.
    void operator()() const { PIKA_TEST(false); }

    template <typename T>
    void operator()(T&& t)
    {
        PIKA_INVOKE(std::move(f), std::forward<T>(t));
    }
};

void check_exception_ptr(std::exception_ptr eptr)
{
    try
    {
        std::rethrow_exception(eptr);
    }
    catch (std::runtime_error const& e)
    {
        PIKA_TEST_EQ(std::string(e.what()), std::string("error"));
    }
}

struct custom_sender_tag_invoke
{
    PIKA_STDEXEC_SENDER_CONCEPT

    std::atomic<bool>& tag_invoke_overload_called;

    template <template <typename...> class Tuple, template <typename...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <typename...> class Variant>
    using error_types = Variant<>;

    static constexpr bool sends_done = false;

    using completion_signatures = pika::execution::experimental::completion_signatures<
        pika::execution::experimental::set_value_t()>;

    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;
        void start() noexcept { pika::execution::experimental::set_value(std::move(r)); }
    };

    template <typename R>
    operation_state<R> connect(R&& r) const
    {
        return {std::forward<R>(r)};
    }
};

struct custom_sender
{
    PIKA_STDEXEC_SENDER_CONCEPT

    std::atomic<bool>& start_called;
    std::atomic<bool>& connect_called;
    std::atomic<bool>& tag_invoke_overload_called;

    template <template <class...> class Tuple, template <class...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    using completion_signatures = pika::execution::experimental::completion_signatures<
        pika::execution::experimental::set_value_t(),
        pika::execution::experimental::set_error_t(std::exception_ptr)>;

    template <typename R>
    struct operation_state
    {
        std::atomic<bool>& start_called;
        std::decay_t<R> r;
        void start() & noexcept
        {
            start_called = true;
            pika::execution::experimental::set_value(std::move(r));
        };
    };

    template <typename R>
    auto connect(R&& r) &&
    {
        connect_called = true;
        return operation_state<R>{start_called, std::forward<R>(r)};
    }
};

template <typename T>
struct custom_typed_sender
{
    PIKA_STDEXEC_SENDER_CONCEPT

    std::decay_t<T> x;

    std::atomic<bool>& start_called;
    std::atomic<bool>& connect_called;
    std::atomic<bool>& tag_invoke_overload_called;

    template <template <class...> class Tuple, template <class...> class Variant>
    using value_types = Variant<Tuple<std::decay_t<T>>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    using completion_signatures = pika::execution::experimental::completion_signatures<
        pika::execution::experimental::set_value_t(std::decay_t<T>),
        pika::execution::experimental::set_error_t(std::exception_ptr)>;

    template <typename R>
    struct operation_state
    {
        std::decay_t<T> x;
        std::atomic<bool>& start_called;
        std::decay_t<R> r;
        void start() & noexcept
        {
            start_called = true;
            pika::execution::experimental::set_value(std::move(r), std::move(x));
        };
    };

    template <typename R>
    auto connect(R&& r) &&
    {
        connect_called = true;
        return operation_state<R>{std::move(x), start_called, std::forward<R>(r)};
    }
};

struct custom_sender2 : custom_sender
{
    explicit custom_sender2(custom_sender s)
      : custom_sender(std::move(s))
    {
    }
};

template <typename T>
struct const_reference_sender
{
    PIKA_STDEXEC_SENDER_CONCEPT

    std::reference_wrapper<std::decay_t<T>> x;

    template <template <class...> class Tuple, template <class...> class Variant>
    using value_types = Variant<Tuple<std::decay_t<T> const&>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    using completion_signatures = pika::execution::experimental::completion_signatures<
        pika::execution::experimental::set_value_t(std::decay_t<T> const&),
        pika::execution::experimental::set_error_t(std::exception_ptr)>;

    template <typename R>
    struct operation_state
    {
        std::reference_wrapper<std::decay_t<T>> const x;
        std::decay_t<R> r;

        void start() & noexcept
        {
            pika::execution::experimental::set_value(std::move(r), x.get());
        };
    };

    template <typename R>
    auto connect(R&& r) &&
    {
        return operation_state<R>{std::move(x), std::forward<R>(r)};
    }

    template <typename R>
    auto connect(R&& r) const&
    {
        return operation_state<R>{x, std::forward<R>(r)};
    }
};

template <typename T>
struct custom_type
{
    std::atomic<bool>& tag_invoke_overload_called;
    std::decay_t<T> x;
};

struct custom_type_non_default_constructible
{
    int x;
    custom_type_non_default_constructible() = delete;
    explicit custom_type_non_default_constructible(int x)
      : x(x){};
    custom_type_non_default_constructible(custom_type_non_default_constructible&&) = default;
    custom_type_non_default_constructible& operator=(
        custom_type_non_default_constructible&&) = default;
    custom_type_non_default_constructible(custom_type_non_default_constructible const&) = default;
    custom_type_non_default_constructible& operator=(
        custom_type_non_default_constructible const&) = default;
};

struct custom_type_non_default_constructible_non_copyable
{
    int x;
    custom_type_non_default_constructible_non_copyable() = delete;
    explicit custom_type_non_default_constructible_non_copyable(int x)
      : x(x){};
    custom_type_non_default_constructible_non_copyable(
        custom_type_non_default_constructible_non_copyable&&) = default;
    custom_type_non_default_constructible_non_copyable& operator=(
        custom_type_non_default_constructible_non_copyable&&) = default;
    custom_type_non_default_constructible_non_copyable(
        custom_type_non_default_constructible_non_copyable const&) = delete;
    custom_type_non_default_constructible_non_copyable& operator=(
        custom_type_non_default_constructible_non_copyable const&) = delete;
};

struct scheduler
{
    std::reference_wrapper<std::atomic<bool>> schedule_called;
    std::reference_wrapper<std::atomic<bool>> execute_called;
    std::reference_wrapper<std::atomic<bool>> tag_invoke_overload_called;

    template <typename F>
    friend void tag_invoke(pika::execution::experimental::execute_t, scheduler s, F&& f)
    {
        s.execute_called.get() = true;
        PIKA_INVOKE(std::forward<F>(f), );
    }

    struct sender
    {
        PIKA_STDEXEC_SENDER_CONCEPT

        std::reference_wrapper<std::atomic<bool>> schedule_called;
        std::reference_wrapper<std::atomic<bool>> execute_called;
        std::reference_wrapper<std::atomic<bool>> tag_invoke_overload_called;

        template <template <class...> class Tuple, template <class...> class Variant>
        using value_types = Variant<Tuple<>>;

        template <template <class...> class Variant>
        using error_types = Variant<std::exception_ptr>;

        static constexpr bool sends_done = false;

        using completion_signatures = pika::execution::experimental::completion_signatures<
            pika::execution::experimental::set_value_t()>;

        template <typename R>
        struct operation_state
        {
            std::decay_t<R> r;

            void start() & noexcept { pika::execution::experimental::set_value(std::move(r)); };
        };

        template <typename R>
        auto connect(R&& r) &&
        {
            return operation_state<R>{std::forward<R>(r)};
        }

        struct env
        {
            std::reference_wrapper<std::atomic<bool>> schedule_called;
            std::reference_wrapper<std::atomic<bool>> execute_called;
            std::reference_wrapper<std::atomic<bool>> tag_invoke_overload_called;

            friend scheduler tag_invoke(pika::execution::experimental::get_completion_scheduler_t<
                                            pika::execution::experimental::set_value_t>,
                env const& e) noexcept
            {
                return {e.schedule_called, e.execute_called, e.tag_invoke_overload_called};
            }
        };

        env get_env() const& noexcept
        {
            return {schedule_called, execute_called, tag_invoke_overload_called};
        }
    };

    friend sender tag_invoke(pika::execution::experimental::schedule_t, scheduler s)
    {
        s.schedule_called.get() = true;
        return {s.schedule_called, s.execute_called, s.tag_invoke_overload_called};
    }

    bool operator==(scheduler const&) const noexcept { return true; }

    bool operator!=(scheduler const&) const noexcept { return false; }
};

struct scheduler2
{
    std::reference_wrapper<std::atomic<bool>> schedule_called;
    std::reference_wrapper<std::atomic<bool>> execute_called;
    std::reference_wrapper<std::atomic<bool>> tag_invoke_overload_called;

    template <typename F>
    friend void tag_invoke(pika::execution::experimental::execute_t, scheduler2 s, F&& f)
    {
        s.execute_called.get() = true;
        PIKA_INVOKE(std::forward<F>(f), );
    }

    struct sender
    {
        PIKA_STDEXEC_SENDER_CONCEPT

        std::reference_wrapper<std::atomic<bool>> schedule_called;
        std::reference_wrapper<std::atomic<bool>> execute_called;
        std::reference_wrapper<std::atomic<bool>> tag_invoke_overload_called;

        template <template <class...> class Tuple, template <class...> class Variant>
        using value_types = Variant<Tuple<>>;

        template <template <class...> class Variant>
        using error_types = Variant<std::exception_ptr>;

        static constexpr bool sends_done = false;

        using completion_signatures = pika::execution::experimental::completion_signatures<
            pika::execution::experimental::set_value_t()>;

        template <typename R>
        struct operation_state
        {
            std::decay_t<R> r;

            void start() & noexcept { pika::execution::experimental::set_value(std::move(r)); };
        };

        template <typename R>
        auto connect(R&& r) &&
        {
            return operation_state<R>{std::forward<R>(r)};
        }

        struct env
        {
            std::reference_wrapper<std::atomic<bool>> schedule_called;
            std::reference_wrapper<std::atomic<bool>> execute_called;
            std::reference_wrapper<std::atomic<bool>> tag_invoke_overload_called;

            friend scheduler2 tag_invoke(pika::execution::experimental::get_completion_scheduler_t<
                                             pika::execution::experimental::set_value_t>,
                env const& e) noexcept
            {
                return {e.schedule_called, e.execute_called, e.tag_invoke_overload_called};
            }
        };

        env get_env() const& noexcept
        {
            return {schedule_called, execute_called, tag_invoke_overload_called};
        }
    };

    friend sender tag_invoke(pika::execution::experimental::schedule_t, scheduler2 s)
    {
        s.schedule_called.get() = true;
        return {s.schedule_called, s.execute_called, s.tag_invoke_overload_called};
    }

    bool operator==(scheduler2 const&) const noexcept { return true; }

    bool operator!=(scheduler2 const&) const noexcept { return false; }
};
