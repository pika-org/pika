//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/allocator_support/internal_allocator.hpp>
#include <pika/assert.hpp>
#include <pika/async_base/launch_policy.hpp>
#include <pika/async_base/traits/is_launch_policy.hpp>
#include <pika/execution/detail/post_policy_dispatch.hpp>
#include <pika/execution/traits/executor_traits.hpp>
#include <pika/execution/traits/future_then_result_exec.hpp>
#include <pika/execution_base/traits/is_executor.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/functional/invoke_result.hpp>
#include <pika/futures/detail/future_data.hpp>
#include <pika/futures/future.hpp>
#include <pika/futures/packaged_continuation.hpp>
#include <pika/futures/traits/future_access.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/memory.hpp>
#include <pika/serialization/detail/polymorphic_nonintrusive_factory.hpp>
#include <pika/timing/steady_clock.hpp>

#include <exception>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

namespace pika { namespace lcos { namespace detail {

    template <typename Executor, typename Future, typename F>
    inline pika::traits::future_then_executor_result_t<Executor,
        std::decay_t<Future>, F>
    then_execute_helper(Executor&& exec, F&& f, Future&& predecessor)
    {
        // simply forward this to executor
        return parallel::execution::then_execute(
            exec, PIKA_FORWARD(F, f), PIKA_FORWARD(Future, predecessor));
    }

    ///////////////////////////////////////////////////////////////////////////
    // launch
    template <typename Future, typename Policy>
    struct future_then_dispatch<Future, Policy,
        std::enable_if_t<traits::is_launch_policy_v<Policy>>>
    {
        template <typename Policy_, typename F>
        PIKA_FORCEINLINE static pika::traits::future_then_result_t<Future, F>
        call(Future&& fut, Policy_&& policy, F&& f)
        {
            using result_type = typename pika::traits::future_then_result<Future,
                F>::result_type;
            using continuation_result_type =
                pika::util::invoke_result_t<F, Future>;

            pika::traits::detail::shared_state_ptr_t<result_type> p =
                detail::make_continuation_alloc<continuation_result_type>(
                    pika::util::internal_allocator<>{}, PIKA_MOVE(fut),
                    PIKA_FORWARD(Policy_, policy), PIKA_FORWARD(F, f));

            return pika::traits::future_access<pika::future<result_type>>::create(
                PIKA_MOVE(p));
        }

        template <typename Allocator, typename Policy_, typename F>
        PIKA_FORCEINLINE static pika::traits::future_then_result_t<Future, F>
        call_alloc(
            Allocator const& alloc, Future&& fut, Policy_&& policy, F&& f)
        {
            using result_type = typename pika::traits::future_then_result<Future,
                F>::result_type;
            using continuation_result_type =
                pika::util::invoke_result_t<F, Future>;

            pika::traits::detail::shared_state_ptr_t<result_type> p =
                detail::make_continuation_alloc<continuation_result_type>(alloc,
                    PIKA_MOVE(fut), PIKA_FORWARD(Policy_, policy),
                    PIKA_FORWARD(F, f));

            return pika::traits::future_access<pika::future<result_type>>::create(
                PIKA_MOVE(p));
        }
    };

    // The overload for future::then taking an executor simply forwards to the
    // corresponding executor customization point.
    //
    // parallel executors v2
    // threads::executor
    template <typename Future, typename Executor>
    struct future_then_dispatch<Future, Executor,
        std::enable_if_t<traits::is_one_way_executor_v<Executor> ||
            traits::is_two_way_executor_v<Executor>>>
    {
        template <typename Executor_, typename F>
        PIKA_FORCEINLINE static pika::traits::future_then_executor_result_t<
            Executor_, Future, F>
        call(Future&& fut, Executor_&& exec, F&& f)
        {
            // simply forward this to executor
            return detail::then_execute_helper(
                PIKA_FORWARD(Executor_, exec), PIKA_FORWARD(F, f), PIKA_MOVE(fut));
        }

        template <typename Allocator, typename Executor_, typename F>
        PIKA_FORCEINLINE static pika::traits::future_then_executor_result_t<
            Executor_, Future, F>
        call_alloc(Allocator const&, Future&& fut, Executor_&& exec, F&& f)
        {
            return call(PIKA_FORWARD(Future, fut), PIKA_FORWARD(Executor_, exec),
                PIKA_FORWARD(F, f));
        }
    };

    // plain function, or function object
    template <typename Future, typename FD>
    struct future_then_dispatch<Future, FD,
        std::enable_if_t<!traits::is_launch_policy_v<FD> &&
            !(traits::is_one_way_executor_v<FD> ||
                traits::is_two_way_executor_v<FD>)>>
    {
        template <typename F>
        PIKA_FORCEINLINE static auto call(Future&& fut, F&& f)
            -> decltype(future_then_dispatch<Future, launch>::call(
                PIKA_MOVE(fut), launch::all, PIKA_FORWARD(F, f)))
        {
            return future_then_dispatch<Future, launch>::call(
                PIKA_MOVE(fut), launch::all, PIKA_FORWARD(F, f));
        }

        template <typename Allocator, typename F>
        PIKA_FORCEINLINE static auto call_alloc(
            Allocator const& alloc, Future&& fut, F&& f)
            -> decltype(future_then_dispatch<Future, launch>::call_alloc(
                alloc, PIKA_MOVE(fut), launch::all, PIKA_FORWARD(F, f)))
        {
            return future_then_dispatch<Future, launch>::call_alloc(
                alloc, PIKA_MOVE(fut), launch::all, PIKA_FORWARD(F, f));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    struct post_policy_spawner
    {
        template <typename F>
        void operator()(F&& f, pika::util::thread_description desc)
        {
            parallel::execution::detail::post_policy_dispatch<
                pika::launch::async_policy>::call(pika::launch::async, desc,
                PIKA_FORWARD(F, f));
        }
    };

    template <typename Executor>
    struct executor_spawner
    {
        Executor exec;

        template <typename F>
        void operator()(F&& f, pika::util::thread_description)
        {
            pika::parallel::execution::post(exec, PIKA_FORWARD(F, f));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename ContResult, typename Future, typename Policy, typename F>
    inline traits::detail::shared_state_ptr_t<continuation_result_t<ContResult>>
    make_continuation(Future const& future, Policy&& policy, F&& f)
    {
        using result_type = continuation_result_t<ContResult>;
        using shared_state = detail::continuation<Future, F, result_type>;
        using init_no_addref = typename shared_state::init_no_addref;
        using spawner_type = post_policy_spawner;

        // create a continuation
        traits::detail::shared_state_ptr_t<result_type> p(
            new shared_state(init_no_addref{}, PIKA_FORWARD(F, f)), false);

        static_cast<shared_state*>(p.get())->template attach<spawner_type>(
            future, spawner_type{}, PIKA_FORWARD(Policy, policy));

        return p;
    }

    // same as above, except with allocator
    template <typename ContResult, typename Allocator, typename Future,
        typename Policy, typename F>
    inline traits::detail::shared_state_ptr_t<continuation_result_t<ContResult>>
    make_continuation_alloc(
        Allocator const& a, Future const& future, Policy&& policy, F&& f)
    {
        using result_type = continuation_result_t<ContResult>;

        using base_allocator = Allocator;
        using shared_state = traits::shared_state_allocator_t<
            detail::continuation<Future, F, result_type>, base_allocator>;

        using other_allocator = typename std::allocator_traits<
            base_allocator>::template rebind_alloc<shared_state>;
        using traits = std::allocator_traits<other_allocator>;

        using init_no_addref = typename shared_state::init_no_addref;

        using unique_ptr = std::unique_ptr<shared_state,
            util::allocator_deleter<other_allocator>>;

        using spawner_type = post_policy_spawner;

        other_allocator alloc(a);
        unique_ptr p(traits::allocate(alloc, 1),
            util::allocator_deleter<other_allocator>{alloc});
        traits::construct(
            alloc, p.get(), init_no_addref{}, alloc, PIKA_FORWARD(F, f));

        // create a continuation
        pika::traits::detail::shared_state_ptr_t<result_type> r(
            p.release(), false);

        static_cast<shared_state*>(r.get())->template attach<spawner_type>(
            future, spawner_type{}, PIKA_FORWARD(Policy, policy));

        return r;
    }

    // same as above, except with allocator and without unwrapping returned
    // futures
    template <typename ContResult, typename Allocator, typename Future,
        typename Policy, typename F>
    inline traits::detail::shared_state_ptr_t<ContResult>
    make_continuation_alloc_nounwrap(
        Allocator const& a, Future const& future, Policy&& policy, F&& f)
    {
        using result_type = ContResult;

        using base_allocator = Allocator;
        using shared_state = traits::shared_state_allocator_t<
            detail::continuation<Future, F, result_type>, base_allocator>;

        using other_allocator = typename std::allocator_traits<
            base_allocator>::template rebind_alloc<shared_state>;
        using traits = std::allocator_traits<other_allocator>;

        using init_no_addref = typename shared_state::init_no_addref;

        using unique_ptr = std::unique_ptr<shared_state,
            util::allocator_deleter<other_allocator>>;

        using spawner_type = post_policy_spawner;

        other_allocator alloc(a);
        unique_ptr p(traits::allocate(alloc, 1),
            util::allocator_deleter<other_allocator>{alloc});
        traits::construct(
            alloc, p.get(), init_no_addref{}, alloc, PIKA_FORWARD(F, f));

        // create a continuation
        typename pika::traits::detail::shared_state_ptr<result_type>::type r(
            p.release(), false);

        static_cast<shared_state*>(r.get())
            ->template attach_nounwrap<spawner_type>(
                future, spawner_type{}, PIKA_FORWARD(Policy, policy));

        return r;
    }

    template <typename ContResult, typename Future, typename Executor,
        typename F>
    inline traits::detail::shared_state_ptr_t<ContResult>
    make_continuation_exec(Future const& future, Executor&& exec, F&& f)
    {
        using shared_state = detail::continuation<Future, F, ContResult>;
        using init_no_addref = typename shared_state::init_no_addref;
        using spawner_type = executor_spawner<std::decay_t<Executor>>;

        // create a continuation
        traits::detail::shared_state_ptr_t<ContResult> p(
            new shared_state(init_no_addref{}, PIKA_FORWARD(F, f)), false);

        static_cast<shared_state*>(p.get())
            ->template attach_nounwrap<spawner_type>(future,
                spawner_type{PIKA_FORWARD(Executor, exec)},
                launch::async_policy{});

        return p;
    }

    template <typename ContResult, typename Future, typename Executor,
        typename Policy, typename F>
    inline traits::detail::shared_state_ptr_t<ContResult>
    make_continuation_exec_policy(
        Future const& future, Executor&& exec, Policy&& policy, F&& f)
    {
        using shared_state = detail::continuation<Future, F, ContResult>;
        using init_no_addref = typename shared_state::init_no_addref;
        using spawner_type = executor_spawner<std::decay_t<Executor>>;

        // create a continuation
        traits::detail::shared_state_ptr_t<ContResult> p(
            new shared_state(init_no_addref{}, PIKA_FORWARD(F, f)), false);

        static_cast<shared_state*>(p.get())
            ->template attach_nounwrap<spawner_type>(future,
                spawner_type{PIKA_FORWARD(Executor, exec)},
                PIKA_FORWARD(Policy, policy));

        return p;
    }
}}}    // namespace pika::lcos::detail
