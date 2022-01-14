//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/async_base/launch_policy.hpp>
#include <pika/functional/detail/invoke.hpp>
#include <pika/functional/traits/is_action.hpp>
#include <pika/futures/future.hpp>
#include <pika/futures/futures_factory.hpp>

#include <functional>
#include <type_traits>
#include <utility>

namespace pika { namespace detail {

    // dispatch point used for launch_policy implementations
    template <typename Action, typename Enable = void>
    struct sync_launch_policy_dispatch;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct sync_launch_policy_dispatch<Action,
        std::enable_if_t<!traits::is_action_v<Action>>>
    {
        // general case execute on separate thread (except launch::sync)
        template <typename F, typename... Ts>
        PIKA_FORCEINLINE static pika::util::detail::invoke_deferred_result_t<F,
            Ts...>
        call(launch policy, F&& f, Ts&&... ts)
        {
            using result_type =
                util::detail::invoke_deferred_result_t<F, Ts...>;

            if (policy == launch::sync)
            {
                return call(
                    launch::sync, PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
            }

            lcos::local::futures_factory<result_type()> p(
                util::deferred_call(PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...));

            if (pika::detail::has_async_policy(policy))
            {
                threads::thread_id_ref_type tid =
                    p.apply(policy, policy.priority());
                if (tid && policy == launch::fork)
                {
                    // make sure this thread is executed last
                    // yield_to
                    pika::this_thread::suspend(
                        threads::thread_schedule_state::pending, tid.noref(),
                        "sync_launch_policy_dispatch<fork>");
                }
            }

            return p.get_future().get();
        }

        // launch::sync execute inline
        template <typename F, typename... Ts>
        PIKA_FORCEINLINE static pika::util::detail::invoke_deferred_result_t<F,
            Ts...>
        call(launch::sync_policy, F&& f, Ts&&... ts)
        {
            try
            {
                return PIKA_INVOKE(PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
            }
            catch (std::bad_alloc const& ba)
            {
                throw ba;
            }
            catch (...)
            {
                throw exception_list(std::current_exception());
            }
        }
    };
}}    // namespace pika::detail
