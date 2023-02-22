//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution_base/receiver.hpp>
#include <pika/testing.hpp>

#include <exception>
#include <string>
#include <type_traits>
#include <utility>

namespace ex = pika::execution::experimental;

bool done_called = false;
bool error_called = false;
bool value_called = false;

namespace mylib {
    struct receiver_1
    {
        friend void tag_invoke(ex::set_stopped_t, receiver_1&&) noexcept
        {
            done_called = true;
        }

        friend void tag_invoke(ex::set_error_t, receiver_1&&, std::exception_ptr) noexcept
        {
            error_called = true;
        }

        friend void tag_invoke(ex::set_value_t, receiver_1&&, int) noexcept
        {
            value_called = true;
        }

        friend constexpr ex::detail::empty_env tag_invoke(ex::get_env_t, receiver_1 const&) noexcept
        {
            return {};
        }
    };

    struct receiver_2
    {
        friend void tag_invoke(ex::set_stopped_t, receiver_2&&) noexcept
        {
            done_called = true;
        }

        friend void tag_invoke(ex::set_error_t, receiver_2&&, int) noexcept
        {
            error_called = true;
        }

        friend constexpr ex::detail::empty_env tag_invoke(ex::get_env_t, receiver_2 const&) noexcept
        {
            return {};
        }
    };

    struct receiver_3
    {
        friend void tag_invoke(ex::set_stopped_t, receiver_3&&) noexcept
        {
            done_called = true;
        }

        friend void tag_invoke(ex::set_error_t, receiver_3&&, std::exception_ptr) noexcept
        {
            error_called = true;
        }

        friend void tag_invoke(ex::set_value_t, receiver_3, int) noexcept
        {
            value_called = true;
        }

        friend constexpr ex::detail::empty_env tag_invoke(ex::get_env_t, receiver_3 const&) noexcept
        {
            return {};
        }
    };

    struct non_receiver_1
    {
        friend void tag_invoke(ex::set_stopped_t, non_receiver_1&) noexcept
        {
            done_called = true;
        }

        friend void tag_invoke(ex::set_error_t, non_receiver_1&&, std::exception_ptr) noexcept
        {
            error_called = true;
        }

        friend void tag_invoke(ex::set_value_t, non_receiver_1, int) noexcept
        {
            value_called = true;
        }
    };

    struct non_receiver_2
    {
        friend void tag_invoke(ex::set_stopped_t, non_receiver_2&&) noexcept
        {
            done_called = true;
        }

        friend void tag_invoke(ex::set_error_t, non_receiver_2&, std::exception_ptr) noexcept
        {
            error_called = true;
        }

        friend void tag_invoke(ex::set_value_t, non_receiver_2, int) noexcept
        {
            value_called = true;
        }
    };

    struct non_receiver_3
    {
        friend void tag_invoke(ex::set_stopped_t, non_receiver_3&) noexcept
        {
            done_called = true;
        }

        friend void tag_invoke(ex::set_error_t, non_receiver_3&, std::exception_ptr) noexcept
        {
            error_called = true;
        }

        friend void tag_invoke(ex::set_value_t, non_receiver_3, int) noexcept
        {
            value_called = true;
        }
    };

    struct non_receiver_4
    {
        friend void tag_invoke(ex::set_stopped_t, non_receiver_4&&) noexcept
        {
            done_called = true;
        }

        friend void tag_invoke(ex::set_error_t, non_receiver_4&&, std::exception_ptr) noexcept
        {
            error_called = true;
        }

        friend void tag_invoke(ex::set_value_t, non_receiver_4&, int) noexcept
        {
            value_called = true;
        }

        friend constexpr ex::detail::empty_env tag_invoke(
            ex::get_env_t, non_receiver_4 const&) noexcept
        {
            return {};
        }
    };

    struct non_receiver_5
    {
        friend void tag_invoke(ex::set_stopped_t, non_receiver_5&&)
        {
            done_called = true;
        }

        friend void tag_invoke(ex::set_error_t, non_receiver_5&&, std::exception_ptr) noexcept
        {
            error_called = true;
        }
    };

    struct non_receiver_6
    {
        friend void tag_invoke(ex::set_stopped_t, non_receiver_6&&) noexcept
        {
            done_called = true;
        }

        friend void tag_invoke(ex::set_error_t, non_receiver_6&&, std::exception_ptr)
        {
            error_called = true;
        }
    };

    struct non_receiver_7
    {
        friend void tag_invoke(ex::set_stopped_t, non_receiver_7&&)
        {
            done_called = true;
        }

        friend void tag_invoke(ex::set_error_t, non_receiver_7&&, std::exception_ptr)
        {
            error_called = true;
        }
    };
}    // namespace mylib

// nvc++ fails on the receiver_of concept.
#if !defined(PIKA_NVHPC_VERSION) || !defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
// Hide differences between receiver_of in the reference implementation (which
// takes a set of completion signatures) and our implementation (which only
// takes a single value type and an optional error type).
template <typename Receiver, typename... Ts>
struct receiver_of_helper
{
#if defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
    static constexpr bool value =
        ex::receiver_of<Receiver, ex::completion_signatures<ex::set_value_t(Ts...)>>;
#else
    static constexpr bool value = ex::is_receiver_of_v<Receiver, Ts...>;
#endif
};

template <typename Receiver, typename... Ts>
inline constexpr bool receiver_of_helper_v = receiver_of_helper<Receiver, Ts...>::value;
#endif

template <typename Receiver, typename Error = std::exception_ptr>
struct receiver_helper
{
#if defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
    static constexpr bool value = ex::receiver<Receiver>;
#else
    static constexpr bool value = ex::is_receiver_v<Receiver, Error>;
#endif
};

template <typename Receiver, typename Error = std::exception_ptr>
inline constexpr bool receiver_helper_v = receiver_helper<Receiver, Error>::value;

int main()
{
    static_assert(ex::is_receiver_v<mylib::receiver_1>, "mylib::receiver_1 should be a receiver");
#if !defined(PIKA_NVHPC_VERSION) || !defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
    static_assert(receiver_of_helper_v<mylib::receiver_1, int>,
        "mylib::receiver_1 should be a receiver of an int");
    static_assert(!receiver_of_helper_v<mylib::receiver_1, std::string>,
        "mylib::receiver_1 should not be a receiver of a std::string");
#endif

#if defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
    static_assert(receiver_helper_v<mylib::receiver_2>, "mylib::receiver_2 should be a receiver");
#else
    // This implicitly checks if the receiver has a set_error std::exception_ptr
    // overload.
    static_assert(!receiver_helper_v<mylib::receiver_2>,
        "mylib::receiver_2 should not be a receiver of std::exception_ptr");
    static_assert(
        receiver_helper_v<mylib::receiver_2, int>, "mylib::receiver_2 should be a receiver");
#endif
#if !defined(PIKA_NVHPC_VERSION) || !defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
    static_assert(!receiver_of_helper_v<mylib::receiver_2, int>,
        "mylib::receiver_2 should not be a receiver of int");
#endif

    static_assert(ex::is_receiver_v<mylib::receiver_3>, "mylib::receiver_3 should be a receiver");
#if !defined(PIKA_NVHPC_VERSION) || !defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
    static_assert(receiver_of_helper_v<mylib::receiver_3, int>,
        "mylib::receiver_3 should be a receiver of an int");
    static_assert(!receiver_of_helper_v<mylib::receiver_3, std::string>,
        "mylib::receiver_3 should not be a receiver of a std::string");
#endif

    static_assert(!ex::is_receiver_v<mylib::non_receiver_1>,
        "mylib::non_receiver_1 should not be a receiver");
    static_assert(!ex::is_receiver_v<mylib::non_receiver_2>,
        "mylib::non_receiver_2 should not be a receiver");
    static_assert(!ex::is_receiver_v<mylib::non_receiver_3>,
        "mylib::non_receiver_3 should not be a receiver");
    static_assert(!receiver_of_helper_v<mylib::non_receiver_3, int>,
        "mylib::non_receiver_3 should not be a receiver of int");
    static_assert(
        ex::is_receiver_v<mylib::non_receiver_4>, "mylib::non_receiver_4 should be a receiver");
    static_assert(!receiver_of_helper_v<mylib::non_receiver_4, int>,
        "mylib::non_receiver_4 should not be a receiver of int");
    static_assert(!ex::is_receiver_v<mylib::non_receiver_5>,
        "mylib::non_receiver_5 should not be a receiver");
    static_assert(!ex::is_receiver_v<mylib::non_receiver_6>,
        "mylib::non_receiver_6 should not be a receiver");
    static_assert(!ex::is_receiver_v<mylib::non_receiver_7>,
        "mylib::non_receiver_7 should not be a receiver");

#if !defined(PIKA_NVHPC_VERSION) || !defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
    static_assert(!receiver_of_helper_v<mylib::non_receiver_1, int>,
        "mylib::non_receiver_1 should not be a receiver of int");
    static_assert(!receiver_of_helper_v<mylib::non_receiver_2, int>,
        "mylib::non_receiver_2 should not be a receiver of int");
    static_assert(!receiver_of_helper_v<mylib::non_receiver_3, int>,
        "mylib::non_receiver_3 should not be a receiver of int");
    static_assert(!receiver_of_helper_v<mylib::non_receiver_4, int>,
        "mylib::non_receiver_4 should not be a receiver of int");
    static_assert(!receiver_of_helper_v<mylib::non_receiver_5, int>,
        "mylib::non_receiver_5 should not be a receiver of int");
    static_assert(!receiver_of_helper_v<mylib::non_receiver_6, int>,
        "mylib::non_receiver_6 should not be a receiver of int");
    static_assert(!receiver_of_helper_v<mylib::non_receiver_7, int>,
        "mylib::non_receiver_7 should not be a receiver of int");
#endif

    {
        mylib::receiver_1 rcv;
        ex::set_stopped(std::move(rcv));
        PIKA_TEST(done_called);
        done_called = false;
    }
    {
        mylib::receiver_1 rcv;
        ex::set_error(std::move(rcv), std::exception_ptr{});
        PIKA_TEST(error_called);
        error_called = false;
    }
    {
        mylib::receiver_1 rcv;
        ex::set_value(std::move(rcv), 1);
        PIKA_TEST(value_called);
        value_called = false;
    }
    {
        mylib::receiver_2 rcv;
        ex::set_stopped(std::move(rcv));
        PIKA_TEST(done_called);
        done_called = false;
    }
    {
        mylib::receiver_2 rcv;
        ex::set_error(std::move(rcv), 1);
        PIKA_TEST(error_called);
        error_called = false;
    }

    return 0;
}
