//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution/scheduler_queries.hpp>
#include <pika/testing.hpp>

#include <exception>
#include <utility>

namespace ex = pika::execution::experimental;

struct uncustomized_scheduler
{
    struct sender
    {
        template <template <class...> class Tuple, template <class...> class Variant>
        using value_types = Variant<Tuple<>>;

        template <template <class...> class Variant>
        using error_types = Variant<std::exception_ptr>;

        static constexpr bool sends_done = false;

        template <typename R>
        struct operation_state
        {
            std::decay_t<R> r;

            void start() & noexcept { ex::set_value(std::move(r)); };
        };

        template <typename R>
        friend auto tag_invoke(ex::connect_t, sender&&, R&& r)
        {
            return operation_state<R>{std::forward<R>(r)};
        }
    };

    friend sender tag_invoke(ex::schedule_t, uncustomized_scheduler) { return {}; }

    bool operator==(uncustomized_scheduler const&) const noexcept { return true; }

    bool operator!=(uncustomized_scheduler const&) const noexcept { return false; }
};

struct customized_scheduler : uncustomized_scheduler
{
    friend ex::forward_progress_guarantee tag_invoke(
        ex::get_forward_progress_guarantee_t, customized_scheduler) noexcept
    {
        return ex::forward_progress_guarantee::concurrent;
    }
};

inline constexpr struct custom_scheduler_query_t
{
    friend constexpr bool tag_invoke(
#if defined(PIKA_HAVE_STDEXEC)
        ex::forwarding_query_t, custom_scheduler_query_t const&) noexcept
#else
        ex::forwarding_scheduler_query_t, custom_scheduler_query_t const&) noexcept
#endif
    {
        return true;
    }
} custom_scheduler_query{};

int main()
{
    // An uncustomized scheduler has weakly_parallel forward progress guarantees
    {
        PIKA_TEST(ex::get_forward_progress_guarantee(uncustomized_scheduler{}) ==
            ex::forward_progress_guarantee::weakly_parallel);
    }

    // A customized scheduler can have other forward progress guarantees
    {
        PIKA_TEST(ex::get_forward_progress_guarantee(customized_scheduler{}) ==
            ex::forward_progress_guarantee::concurrent);
    }

    // get_forward_progress_guarantee is not a forwarding query
    {
#if defined(PIKA_HAVE_STDEXEC)
        static_assert(!ex::forwarding_query(ex::get_forward_progress_guarantee));
#else
        static_assert(!ex::forwarding_scheduler_query(ex::get_forward_progress_guarantee));
#endif
    }

    // The custom query is a forwarding query
    {
#if defined(PIKA_HAVE_STDEXEC)
        static_assert(ex::forwarding_query(custom_scheduler_query));
#else
        static_assert(ex::forwarding_scheduler_query(custom_scheduler_query));
#endif
    }
}
