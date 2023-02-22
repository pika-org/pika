//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution/algorithms/execute.hpp>
#include <pika/testing.hpp>

#include <cstddef>
#include <exception>
#include <type_traits>

static std::size_t friend_tag_invoke_schedule_calls = 0;
static std::size_t tag_invoke_execute_calls = 0;

template <typename Scheduler>
struct sender
{
    template <template <class...> class Tuple, template <class...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    using completion_signatures = pika::execution::experimental::completion_signatures<
        pika::execution::experimental::set_value_t()>;

    struct operation_state
    {
        friend void tag_invoke(
            pika::execution::experimental::start_t, operation_state&) noexcept {};
    };

    template <typename R>
    friend operation_state
    tag_invoke(pika::execution::experimental::connect_t, sender&&, R&&) noexcept
    {
        return {};
    }

    friend Scheduler tag_invoke(pika::execution::experimental::get_completion_scheduler_t<
                                    pika::execution::experimental::set_value_t>,
        sender const&) noexcept
    {
        return {};
    }
};

struct scheduler_1
{
    friend sender<scheduler_1> tag_invoke(pika::execution::experimental::schedule_t, scheduler_1)
    {
        ++friend_tag_invoke_schedule_calls;
        return {};
    }

    bool operator==(scheduler_1 const&) const noexcept
    {
        return true;
    }

    bool operator!=(scheduler_1 const&) const noexcept
    {
        return false;
    }
};

struct scheduler_2
{
    friend sender<scheduler_2> tag_invoke(pika::execution::experimental::schedule_t, scheduler_2)
    {
        PIKA_TEST(false);
        return {};
    }

    bool operator==(scheduler_2 const&) const noexcept
    {
        return true;
    }

    bool operator!=(scheduler_2 const&) const noexcept
    {
        return false;
    }
};

template <typename F>
void tag_invoke(pika::execution::experimental::execute_t, scheduler_2, F&&)
{
    ++tag_invoke_execute_calls;
}

struct f_struct_1
{
    void operator()() noexcept {};
};

struct f_struct_2
{
    void operator()(int = 42){};
};

void f_fun_1() {}

void f_fun_2(int) {}

int main()
{
    {
        scheduler_1 s1;
        pika::execution::experimental::execute(s1, f_struct_1{});
        pika::execution::experimental::execute(s1, f_struct_2{});
        pika::execution::experimental::execute(s1, &f_fun_1);
        PIKA_TEST_EQ(friend_tag_invoke_schedule_calls, std::size_t(3));
        PIKA_TEST_EQ(tag_invoke_execute_calls, std::size_t(0));
    }

    {
        scheduler_2 s2;
        pika::execution::experimental::execute(s2, f_struct_1{});
        pika::execution::experimental::execute(s2, f_struct_2{});
        pika::execution::experimental::execute(s2, &f_fun_1);
        PIKA_TEST_EQ(friend_tag_invoke_schedule_calls, std::size_t(3));
        PIKA_TEST_EQ(tag_invoke_execute_calls, std::size_t(3));
    }

    return 0;
}
