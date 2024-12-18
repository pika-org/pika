//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution_base/completion_scheduler.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/testing.hpp>

#include <cstddef>
#include <exception>
#include <type_traits>

namespace ex = pika::execution::experimental;

static std::size_t friend_tag_invoke_schedule_calls = 0;
static std::size_t tag_invoke_schedule_calls = 0;

template <typename Scheduler>
struct sender
{
    PIKA_STDEXEC_SENDER_CONCEPT

    template <template <class...> class Tuple, template <class...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    using completion_signatures =
        ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr)>;

    struct operation_state
    {
        friend operation_state tag_invoke(ex::start_t, operation_state&) noexcept { return {}; }
    };

    template <typename R>
    friend operation_state tag_invoke(ex::connect_t, sender, R&&) noexcept
    {
        return {};
    }

    struct env
    {
        friend Scheduler tag_invoke(
            ex::get_completion_scheduler_t<ex::set_value_t>, env const&) noexcept
        {
            return {};
        }
    };

    friend env tag_invoke(ex::get_env_t, sender const&) noexcept { return {}; }
};

struct non_scheduler_1
{
};

struct non_scheduler_2
{
    void schedule() {}
};

struct non_scheduler_3
{
    friend sender<non_scheduler_3> tag_invoke(ex::schedule_t, non_scheduler_3) { return {}; }
};

struct scheduler_1
{
    friend sender<scheduler_1> tag_invoke(ex::schedule_t const&, scheduler_1)
    {
        ++friend_tag_invoke_schedule_calls;
        return {};
    }

    bool operator==(scheduler_1 const&) const noexcept { return true; }

    bool operator!=(scheduler_1 const&) const noexcept { return false; }
};

struct scheduler_2
{
    bool operator==(scheduler_2 const&) const noexcept { return true; }

    bool operator!=(scheduler_2 const&) const noexcept { return false; }
};

sender<scheduler_2> tag_invoke(ex::schedule_t, scheduler_2)
{
    ++tag_invoke_schedule_calls;
    return {};
}

int main()
{
    static_assert(!ex::is_scheduler_v<non_scheduler_1>, "non_scheduler_1 is not a scheduler");
    // stdexec static_asserts that a member schedule must return a sender
#if !defined(PIKA_HAVE_STDEXEC)
    static_assert(!ex::is_scheduler_v<non_scheduler_2>, "non_scheduler_2 is not a scheduler");
#endif
    static_assert(!ex::is_scheduler_v<non_scheduler_3>, "non_scheduler_3 is not a scheduler");
    static_assert(ex::is_scheduler_v<scheduler_1>, "scheduler_1 is a scheduler");
    static_assert(ex::is_scheduler_v<scheduler_2>, "scheduler_2 is a scheduler");

    scheduler_1 s1;
    [[maybe_unused]] auto snd1 = ex::schedule(s1);
    PIKA_TEST_EQ(friend_tag_invoke_schedule_calls, std::size_t(1));
    PIKA_TEST_EQ(tag_invoke_schedule_calls, std::size_t(0));

    scheduler_2 s2;
    [[maybe_unused]] auto snd2 = ex::schedule(s2);
    PIKA_TEST_EQ(friend_tag_invoke_schedule_calls, std::size_t(1));
    PIKA_TEST_EQ(tag_invoke_schedule_calls, std::size_t(1));

    return 0;
}
