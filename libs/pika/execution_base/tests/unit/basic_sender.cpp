//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/testing.hpp>

#include <cstddef>
#include <exception>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>

namespace ex = pika::execution::experimental;

static std::size_t friend_tag_invoke_connect_calls = 0;
static std::size_t tag_invoke_connect_calls = 0;

struct non_sender_1
{
};

struct non_sender_2
{
    template <template <class...> class Variant>
    using error_types = Variant<>;

    static constexpr bool sends_done = false;
};

struct non_sender_3
{
    template <template <class...> class Tuple, template <class...> class Variant>
    using value_types = Variant<Tuple<>>;

    static constexpr bool sends_done = false;
};

struct non_sender_4
{
    template <template <class...> class Tuple, template <class...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <class...> class Variant>
    using error_types = Variant<>;
};

struct non_sender_5
{
    static constexpr bool sends_done = false;
};

struct non_sender_6
{
    template <template <class...> class Variant>
    using error_types = Variant<>;
};

struct non_sender_7
{
    template <template <class...> class Tuple, template <class...> class Variant>
    using value_types = Variant<Tuple<>>;
};

struct receiver
{
    PIKA_STDEXEC_RECEIVER_CONCEPT

    void set_error(std::exception_ptr) && noexcept {}

    friend void tag_invoke(ex::set_stopped_t, receiver&&) noexcept {}

    void set_value(int v) && noexcept { i.get() = v; }

    friend constexpr ex::empty_env tag_invoke(ex::get_env_t, receiver const&) noexcept
    {
        return {};
    }

    std::reference_wrapper<int> i;
};

struct sender_1
{
    PIKA_STDEXEC_SENDER_CONCEPT

    template <template <class...> class Tuple, template <class...> class Variant>
    using value_types = Variant<Tuple<int>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    using completion_signatures =
        ex::completion_signatures<ex::set_value_t(int), ex::set_error_t(std::exception_ptr)>;

    struct operation_state
    {
        receiver r;
        friend void tag_invoke(ex::start_t, operation_state& os) noexcept
        {
            ex::set_value(std::move(os.r), 4711);
        };
    };

    friend operation_state tag_invoke(ex::connect_t, sender_1&&, receiver r)
    {
        ++friend_tag_invoke_connect_calls;
        return {r};
    }
};

struct sender_2
{
    PIKA_STDEXEC_SENDER_CONCEPT

    template <template <class...> class Tuple, template <class...> class Variant>
    using value_types = Variant<Tuple<int>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    using completion_signatures =
        ex::completion_signatures<ex::set_value_t(int), ex::set_error_t(std::exception_ptr)>;

    struct operation_state
    {
        receiver r;
        friend void tag_invoke(ex::start_t, operation_state& os) noexcept
        {
            ex::set_value(std::move(os.r), 4711);
        };
    };
};

sender_2::operation_state tag_invoke(ex::connect_t, sender_2, receiver r)
{
    ++tag_invoke_connect_calls;
    return {r};
}

static std::size_t void_receiver_set_value_calls = 0;

struct void_receiver
{
    PIKA_STDEXEC_RECEIVER_CONCEPT

    void set_error(std::exception_ptr) && noexcept {}

    friend void tag_invoke(ex::set_stopped_t, void_receiver&&) noexcept {}

    void set_value() && noexcept { ++void_receiver_set_value_calls; }

    friend constexpr ex::empty_env tag_invoke(ex::get_env_t, void_receiver const&) noexcept
    {
        return {};
    }
};

#if !defined(PIKA_HAVE_STDEXEC)
template <typename Sender>
constexpr bool unspecialized(...)
{
    return false;
}

template <typename Sender>
constexpr bool unspecialized(typename ex::sender_traits<Sender>::__unspecialized*)
{
    return true;
}
#endif

int main()
{
#if !defined(PIKA_HAVE_STDEXEC)
    static_assert(unspecialized<void>(nullptr), "void should not have sender_traits");
    static_assert(
        unspecialized<std::nullptr_t>(nullptr), "std::nullptr_t should not have sender_traits");
    static_assert(unspecialized<int>(nullptr), "non_sender_1 should not have sender_traits");
    static_assert(unspecialized<double>(nullptr), "non_sender_1 should not have sender_traits");
    static_assert(
        unspecialized<non_sender_1>(nullptr), "non_sender_1 should not have sender_traits");
    static_assert(
        unspecialized<non_sender_2>(nullptr), "non_sender_2 should not have sender_traits");
    static_assert(
        unspecialized<non_sender_3>(nullptr), "non_sender_3 should not have sender_traits");
    static_assert(
        unspecialized<non_sender_4>(nullptr), "non_sender_4 should not have sender_traits");
    static_assert(
        unspecialized<non_sender_5>(nullptr), "non_sender_5 should not have sender_traits");
    static_assert(
        unspecialized<non_sender_6>(nullptr), "non_sender_6 should not have sender_traits");
    static_assert(
        unspecialized<non_sender_7>(nullptr), "non_sender_7 should not have sender_traits");
    static_assert(!unspecialized<sender_1>(nullptr), "sender_1 should have sender_traits");
    static_assert(!unspecialized<sender_2>(nullptr), "sender_2 should have sender_traits");
#endif

    static_assert(!ex::is_sender_v<void>, "void is not a sender");
    static_assert(!ex::is_sender_v<std::nullptr_t>, "std::nullptr_t is not a sender");
    static_assert(!ex::is_sender_v<int>, "int is not a sender");
    static_assert(!ex::is_sender_v<double>, "double is not a sender");
    static_assert(!ex::is_sender_v<non_sender_1>, "non_sender_1 is not a sender");
    static_assert(!ex::is_sender_v<non_sender_2>, "non_sender_2 is not a sender");
    static_assert(!ex::is_sender_v<non_sender_3>, "non_sender_3 is not a sender");
    static_assert(!ex::is_sender_v<non_sender_4>, "non_sender_4 is not a sender");
    static_assert(!ex::is_sender_v<non_sender_5>, "non_sender_5 is not a sender");
    static_assert(!ex::is_sender_v<non_sender_6>, "non_sender_6 is not a sender");
    static_assert(!ex::is_sender_v<non_sender_7>, "non_sender_7 is not a sender");
    static_assert(ex::is_sender_v<sender_1>, "sender_1 is a sender");
    static_assert(ex::is_sender_v<sender_2>, "sender_2 is a sender");

    static_assert(ex::is_sender_to_v<sender_1, receiver>, "sender_1 is a sender to receiver");
    static_assert(ex::is_sender_to_v<sender_1, receiver>, "sender_1 is a sender to receiver");
    static_assert(
        !ex::is_sender_to_v<sender_1, non_sender_1>, "sender_1 is not a sender to non_sender_1");
    static_assert(!ex::is_sender_to_v<sender_1, sender_1>, "sender_1 is not a sender to sender_1");
    static_assert(ex::is_sender_to_v<sender_2, receiver>, "sender_2 is a sender to receiver");
    static_assert(
        !ex::is_sender_to_v<sender_2, non_sender_2>, "sender_2 is not a sender to non_sender_2");
    static_assert(!ex::is_sender_to_v<sender_2, sender_2>, "sender_2 is not a sender to sender_2");

    {
        int i = 1;
        receiver r1{i};
        auto os = ex::connect(sender_1{}, std::move(r1));
        ex::start(os);
        PIKA_TEST_EQ(i, 4711);
        PIKA_TEST_EQ(friend_tag_invoke_connect_calls, std::size_t(1));
        PIKA_TEST_EQ(tag_invoke_connect_calls, std::size_t(0));
    }

    {
        int i = 1;
        receiver r2{i};
        auto os = ex::connect(sender_2{}, std::move(r2));
        ex::start(os);
        PIKA_TEST_EQ(i, 4711);
        PIKA_TEST_EQ(friend_tag_invoke_connect_calls, std::size_t(1));
        PIKA_TEST_EQ(tag_invoke_connect_calls, std::size_t(1));
    }

    return 0;
}
