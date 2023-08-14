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
        friend void tag_invoke(pika::execution::experimental::start_t, operation_state& os) noexcept
        {
            pika::execution::experimental::set_value(std::move(os.r));
        }
    };

    template <typename R>
    friend operation_state<R>
    tag_invoke(pika::execution::experimental::connect_t, void_sender, R&& r)
    {
        return {std::forward<R>(r)};
    }
};

template <typename... Ts>
struct error_sender
{
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
        friend void tag_invoke(pika::execution::experimental::start_t, operation_state& os) noexcept
        {
            try
            {
                throw std::runtime_error("error");
            }
            catch (...)
            {
                pika::execution::experimental::set_error(std::move(os.r), std::current_exception());
            }
        }
    };

    template <typename R>
    friend operation_state<R>
    tag_invoke(pika::execution::experimental::connect_t, error_sender, R&& r)
    {
        return {std::forward<R>(r)};
    }
};

template <typename... Ts>
struct const_reference_error_sender
{
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
        friend void tag_invoke(pika::execution::experimental::start_t, operation_state& os) noexcept
        {
            auto const e = std::make_exception_ptr(std::runtime_error("error"));
            pika::execution::experimental::set_error(std::move(os.r), e);
        }
    };

    template <typename R>
    friend operation_state<R>
    tag_invoke(pika::execution::experimental::connect_t, const_reference_error_sender, R&& r)
    {
        return {std::forward<R>(r)};
    }
};

template <typename F>
struct callback_receiver
{
    std::decay_t<F> f;
    std::atomic<bool>& set_value_called;

    template <typename E>
    friend void
    tag_invoke(pika::execution::experimental::set_error_t, callback_receiver&&, E&&) noexcept
    {
        PIKA_TEST(false);
    }

    friend void tag_invoke(
        pika::execution::experimental::set_stopped_t, callback_receiver&&) noexcept
    {
        PIKA_TEST(false);
    };

    template <typename... Ts>
    friend auto tag_invoke(pika::execution::experimental::set_value_t, callback_receiver&& r,
        Ts&&... ts) noexcept -> decltype(PIKA_INVOKE(f, std::forward<Ts>(ts)...), void())
    {
        PIKA_INVOKE(r.f, std::forward<Ts>(ts)...);
        r.set_value_called = true;
    }

    friend constexpr pika::execution::experimental::empty_env tag_invoke(
        pika::execution::experimental::get_env_t, callback_receiver const&) noexcept
    {
        return {};
    }
};

template <typename F>
struct error_callback_receiver
{
    std::decay_t<F> f;
    std::atomic<bool>& set_error_called;
    bool expect_set_value = false;

    template <typename E>
    friend void tag_invoke(
        pika::execution::experimental::set_error_t, error_callback_receiver&& r, E&& e) noexcept
    {
        PIKA_INVOKE(r.f, std::forward<E>(e));
        r.set_error_called = true;
    }

    friend void tag_invoke(
        pika::execution::experimental::set_stopped_t, error_callback_receiver&&) noexcept
    {
        PIKA_TEST(false);
    };

    template <typename... Ts>
    friend void tag_invoke(
        pika::execution::experimental::set_value_t, error_callback_receiver&& r, Ts&&...) noexcept
    {
        PIKA_TEST(r.expect_set_value);
    }

    friend constexpr pika::execution::experimental::empty_env tag_invoke(
        pika::execution::experimental::get_env_t, error_callback_receiver const&) noexcept
    {
        return {};
    }
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
    catch (const std::runtime_error& e)
    {
        PIKA_TEST_EQ(std::string(e.what()), std::string("error"));
    }
}

struct custom_sender_tag_invoke
{
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
    operation_state<R> connect(R&& r)
    {
        return {std::forward<R>(r)};
    }
};

struct custom_sender
{
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
        friend void tag_invoke(pika::execution::experimental::start_t, operation_state& os) noexcept
        {
            os.start_called = true;
            pika::execution::experimental::set_value(std::move(os.r));
        };
    };

    template <typename R>
    friend auto tag_invoke(pika::execution::experimental::connect_t, custom_sender&& s, R&& r)
    {
        s.connect_called = true;
        return operation_state<R>{s.start_called, std::forward<R>(r)};
    }
};

template <typename T>
struct custom_typed_sender
{
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
        friend void tag_invoke(pika::execution::experimental::start_t, operation_state& os) noexcept
        {
            os.start_called = true;
            pika::execution::experimental::set_value(std::move(os.r), std::move(os.x));
        };
    };

    template <typename R>
    friend auto tag_invoke(pika::execution::experimental::connect_t, custom_typed_sender&& s, R&& r)
    {
        s.connect_called = true;
        return operation_state<R>{std::move(s.x), s.start_called, std::forward<R>(r)};
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

        friend void tag_invoke(pika::execution::experimental::start_t, operation_state& os) noexcept
        {
            pika::execution::experimental::set_value(std::move(os.r), os.x.get());
        };
    };

    template <typename R>
    friend auto
    tag_invoke(pika::execution::experimental::connect_t, const_reference_sender&& s, R&& r)
    {
        return operation_state<R>{std::move(s.x), std::forward<R>(r)};
    }

    template <typename R>
    friend auto
    tag_invoke(pika::execution::experimental::connect_t, const_reference_sender const& s, R&& r)
    {
        return operation_state<R>{s.x, std::forward<R>(r)};
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

            friend void tag_invoke(
                pika::execution::experimental::start_t, operation_state& os) noexcept
            {
                pika::execution::experimental::set_value(std::move(os.r));
            };
        };

        template <typename R>
        friend auto tag_invoke(pika::execution::experimental::connect_t, sender&&, R&& r)
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

        friend env tag_invoke(pika::execution::experimental::get_env_t, sender const& s) noexcept
        {
            return {s.schedule_called, s.execute_called, s.tag_invoke_overload_called};
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

            friend void tag_invoke(
                pika::execution::experimental::start_t, operation_state& os) noexcept
            {
                pika::execution::experimental::set_value(std::move(os.r));
            };
        };

        template <typename R>
        friend auto tag_invoke(pika::execution::experimental::connect_t, sender&&, R&& r)
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

        friend env tag_invoke(pika::execution::experimental::get_env_t, sender const& s) noexcept
        {
            return {s.schedule_called, s.execute_called, s.tag_invoke_overload_called};
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

namespace tag_namespace {
    inline constexpr struct my_tag_t
    {
        template <typename Sender>
        auto operator()(Sender&& sender) const
        {
            return pika::functional::detail::tag_invoke(*this, std::forward<Sender>(sender));
        }

        struct wrapper
        {
            wrapper(my_tag_t) {}
        };

        // This overload should be chosen by test_adl_isolation below. We make
        // sure this is a worse match than the one in my_namespace by requiring
        // a conversion.
        template <typename Sender>
        friend void tag_invoke(wrapper, Sender&&)
        {
        }
    } my_tag{};
}    // namespace tag_namespace

namespace my_namespace {
    // The below types should be used as a template arguments for the sender in
    // test_adl_isolation.
    struct my_type
    {
        void operator()() const {}
        void operator()(int) const {}
        void operator()(std::exception_ptr) const {}
    };

    struct my_scheduler
    {
        struct sender
        {
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

                friend void tag_invoke(
                    pika::execution::experimental::start_t, operation_state& os) noexcept
                {
                    pika::execution::experimental::set_value(std::move(os.r));
                };
            };

            template <typename R>
            friend auto tag_invoke(pika::execution::experimental::connect_t, sender&&, R&& r)
            {
                return operation_state<R>{std::forward<R>(r)};
            }

            friend my_scheduler tag_invoke(
                pika::execution::experimental::get_completion_scheduler_t<
                    pika::execution::experimental::set_value_t>,
                sender const&) noexcept
            {
                return {};
            }

            struct env
            {
                friend my_scheduler tag_invoke(
                    pika::execution::experimental::get_completion_scheduler_t<
                        pika::execution::experimental::set_value_t>,
                    env const&) noexcept
                {
                    return {};
                }
            };

            friend env tag_invoke(pika::execution::experimental::get_env_t, sender const&) noexcept
            {
                return {};
            }
        };

        friend sender tag_invoke(pika::execution::experimental::schedule_t, my_scheduler)
        {
            return {};
        }

        bool operator==(my_scheduler const&) const noexcept { return true; }

        bool operator!=(my_scheduler const&) const noexcept { return false; }
    };

    template <typename... Ts>
    struct my_sender
    {
        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types = Variant<Tuple<Ts...>>;

        template <template <typename...> class Variant>
        using error_types = Variant<>;

        static constexpr bool sends_done = false;

        using completion_signatures = pika::execution::experimental::completion_signatures<
            pika::execution::experimental::set_value_t(Ts...)>;

        template <typename R>
        struct operation_state
        {
            std::decay_t<R> r;
            friend void tag_invoke(
                pika::execution::experimental::start_t, operation_state& os) noexcept
            {
                pika::execution::experimental::set_value(std::move(os.r), Ts{}...);
            }
        };

        template <typename R>
        friend operation_state<R>
        tag_invoke(pika::execution::experimental::connect_t, my_sender, R&& r)
        {
            return {std::forward<R>(r)};
        }
    };

    // This overload should not be chosen by test_adl_isolation below. We make
    // sure this is a better match than the one in tag_namespace so that if this
    // one is visible it is chosen. It should not be visible.
    template <typename Sender>
    void tag_invoke(tag_namespace::my_tag_t, Sender&&)
    {
        static_assert(sizeof(Sender) == 0);
    }
}    // namespace my_namespace

// This test function expects a type that has my_namespace::my_type as a
// template argument. If template arguments are correctly hidden from ADL the
// friend tag_invoke overload in my_tag_t will be chosen. If template arguments
// are not hidden the unconstrained tag_invoke overload in my_namespace will be
// chosen instead.
template <typename Sender>
void test_adl_isolation(Sender&& sender)
{
    tag_namespace::my_tag(std::forward<Sender>(sender));
}
