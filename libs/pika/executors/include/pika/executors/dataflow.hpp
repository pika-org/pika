//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/allocator_support/internal_allocator.hpp>
#include <pika/async_base/dataflow.hpp>
#include <pika/async_base/launch_policy.hpp>
#include <pika/async_base/traits/is_launch_policy.hpp>
#include <pika/coroutines/detail/get_stack_pointer.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/errors/try_catch_exception_ptr.hpp>
#include <pika/execution/executors/execution.hpp>
#include <pika/execution_base/traits/is_executor.hpp>
#include <pika/executors/parallel_executor.hpp>
#include <pika/functional/deferred_call.hpp>
#include <pika/functional/invoke_fused.hpp>
#include <pika/functional/traits/get_function_annotation.hpp>
#include <pika/functional/traits/is_action.hpp>
#include <pika/futures/detail/future_transforms.hpp>
#include <pika/futures/future.hpp>
#include <pika/futures/traits/acquire_future.hpp>
#include <pika/futures/traits/future_access.hpp>
#include <pika/futures/traits/is_future.hpp>
#include <pika/modules/memory.hpp>
#include <pika/pack_traversal/pack_traversal_async.hpp>
#include <pika/threading_base/annotated_function.hpp>
#include <pika/threading_base/thread_num_tss.hpp>
#include <pika/type_support/always_void.hpp>

#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
// forward declare the type we will get function annotations from
namespace pika { namespace lcos { namespace detail {
    template <typename Frame>
    struct dataflow_finalization;
}}}    // namespace pika::lcos::detail

namespace pika { namespace traits {
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
    ///////////////////////////////////////////////////////////////////////////
    // traits specialization to get annotation from dataflow_finalization
    template <typename Frame>
    struct get_function_annotation<lcos::detail::dataflow_finalization<Frame>>
    {
        using function_type = typename Frame::function_type;
        //
        static constexpr char const* call(
            lcos::detail::dataflow_finalization<Frame> const& f) noexcept
        {
            char const* annotation = pika::traits::get_function_annotation<
                typename std::decay<function_type>::type>::call(f.this_->func_);
            return annotation;
        }
    };
#endif
}}    // namespace pika::traits

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace lcos { namespace detail {
    template <typename Frame>
    struct dataflow_finalization
    {
        //
        explicit dataflow_finalization(Frame* df)
          : this_(df)
        {
        }
        using is_void = typename Frame::is_void;
        //
        template <typename Futures>
        void operator()(Futures&& futures) const
        {
            return this_->execute(is_void{}, PIKA_FORWARD(Futures, futures));
        }

        // keep the dataflow frame alive with this pointer reference
        pika::intrusive_ptr<Frame> this_;
    };

    template <typename F, typename Args>
    struct dataflow_not_callable
    {
        static auto error(F f, Args args)
        {
            pika::util::invoke_fused(PIKA_MOVE(f), PIKA_MOVE(args));
        }

        using type = decltype(error(std::declval<F>(), std::declval<Args>()));
    };

    ///////////////////////////////////////////////////////////////////////
    template <bool IsAction, typename Policy, typename F, typename Args,
        typename Enable = void>
    struct dataflow_return_impl
    {
        using type = typename dataflow_not_callable<F, Args>::type;
    };

    template <typename Policy, typename F, typename Args>
    struct dataflow_return_impl<
        /*IsAction=*/false, Policy, F, Args,
        typename std::enable_if<traits::is_launch_policy<Policy>::value>::type>
    {
        using type = pika::future<
            typename util::detail::invoke_fused_result<F, Args>::type>;
    };

    template <typename Executor, typename F, typename Args>
    struct dataflow_return_impl_executor;

    template <typename Executor, typename F, typename... Ts>
    struct dataflow_return_impl_executor<Executor, F, pika::tuple<Ts...>>
    {
        using type = decltype(
            pika::parallel::execution::async_execute(std::declval<Executor&&>(),
                std::declval<F>(), std::declval<Ts>()...));
    };

    template <typename Policy, typename F, typename Args>
    struct dataflow_return_impl<
        /*IsAction=*/false, Policy, F, Args,
        typename std::enable_if<traits::is_one_way_executor<Policy>::value ||
            traits::is_two_way_executor<Policy>::value>::type>
      : dataflow_return_impl_executor<Policy, F, Args>
    {
    };

    template <typename Policy, typename F, typename Args>
    struct dataflow_return
      : detail::dataflow_return_impl<traits::is_action<F>::value, Policy, F,
            Args>
    {
    };

    template <typename Executor, typename Frame, typename Func,
        typename Futures, typename Enable = void>
    struct has_dataflow_finalize : std::false_type
    {
    };

    template <typename Executor, typename Frame, typename Func,
        typename Futures>
    struct has_dataflow_finalize<Executor, Frame, Func, Futures,
        typename pika::util::always_void<decltype(
            std::declval<Executor>().dataflow_finalize(std::declval<Frame>(),
                std::declval<Func>(), std::declval<Futures>()))>::type>
      : std::true_type
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Policy, typename Func, typename Futures>
    struct dataflow_frame    //-V690
      : pika::lcos::detail::future_data<typename pika::traits::future_traits<
            typename detail::dataflow_return<Policy, Func,
                Futures>::type>::type>
    {
        using type =
            typename detail::dataflow_return<Policy, Func, Futures>::type;
        using result_type = typename pika::traits::future_traits<type>::type;
        using base_type = pika::lcos::detail::future_data<result_type>;

        using is_void = std::is_void<result_type>;

        using function_type = Func;
        using dataflow_type = dataflow_frame<Policy, Func, Futures>;

        friend struct dataflow_finalization<dataflow_type>;
        friend struct traits::get_function_annotation<
            dataflow_finalization<dataflow_type>>;

    private:
        // workaround gcc regression wrongly instantiating constructors
        dataflow_frame();
        dataflow_frame(dataflow_frame const&);

    public:
        using init_no_addref = typename base_type::init_no_addref;

        /// A struct to construct the dataflow_frame in-place
        struct construction_data
        {
            Policy policy_;
            Func func_;
        };

        /// Construct the dataflow_frame from the given policy
        /// and callable object.
        static construction_data construct_from(Policy policy, Func func)
        {
            return construction_data{PIKA_MOVE(policy), PIKA_MOVE(func)};
        }

        explicit dataflow_frame(construction_data data)
          : base_type(init_no_addref{})
          , policy_(PIKA_MOVE(data.policy_))
          , func_(PIKA_MOVE(data.func_))
        {
        }

    private:
        ///////////////////////////////////////////////////////////////////////
        /// Passes the futures into the evaluation function and
        /// sets the result future.
        template <typename Futures_>
        PIKA_FORCEINLINE void execute(std::false_type, Futures_&& futures)
        {
            pika::detail::try_catch_exception_ptr(
                [&]() {
                    this->set_data(util::invoke_fused(
                        PIKA_MOVE(func_), PIKA_FORWARD(Futures_, futures)));
                },
                [&](std::exception_ptr ep) {
                    this->set_exception(PIKA_MOVE(ep));
                });
        }

        /// Passes the futures into the evaluation function and
        /// sets the result future.
        template <typename Futures_>
        PIKA_FORCEINLINE void execute(std::true_type, Futures_&& futures)
        {
            pika::detail::try_catch_exception_ptr(
                [&]() {
                    util::invoke_fused(
                        PIKA_MOVE(func_), PIKA_FORWARD(Futures_, futures));

                    this->set_data(util::unused_type());
                },
                [&](std::exception_ptr ep) {
                    this->set_exception(PIKA_MOVE(ep));
                });
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Futures_>
        void finalize(pika::detail::async_policy policy, Futures_&& futures)
        {
            detail::dataflow_finalization<dataflow_type> this_f_(this);

            pika::execution::parallel_policy_executor<launch::async_policy> exec{
                policy};

            exec.post(PIKA_MOVE(this_f_), PIKA_FORWARD(Futures_, futures));
        }

        template <typename Futures_>
        void finalize(pika::detail::fork_policy policy, Futures_&& futures)
        {
            detail::dataflow_finalization<dataflow_type> this_f_(this);

            pika::execution::parallel_policy_executor<launch::fork_policy> exec{
                policy};

            exec.post(PIKA_MOVE(this_f_), PIKA_FORWARD(Futures_, futures));
        }

        template <typename Futures_>
        PIKA_FORCEINLINE void finalize(
            pika::detail::sync_policy, Futures_&& futures)
        {
            // We need to run the completion on a new thread if we are on a
            // non pika thread.
            bool recurse_asynchronously =
                pika::threads::get_self_ptr() == nullptr;
#if defined(PIKA_HAVE_THREADS_GET_STACK_POINTER)
            recurse_asynchronously = !this_thread::has_sufficient_stack_space();
#else
            struct handle_continuation_recursion_count
            {
                handle_continuation_recursion_count()
                  : count_(threads::get_continuation_recursion_count())
                {
                    ++count_;
                }
                ~handle_continuation_recursion_count()
                {
                    --count_;
                }

                std::size_t& count_;
            } cnt;
            recurse_asynchronously = recurse_asynchronously ||
                cnt.count_ > PIKA_CONTINUATION_MAX_RECURSION_DEPTH;
#endif
            if (!recurse_asynchronously)
            {
                pika::scoped_annotation annotate(func_);
                execute(is_void{}, PIKA_FORWARD(Futures_, futures));
            }
            else
            {
                finalize(pika::launch::async, PIKA_FORWARD(Futures_, futures));
            }
        }

        template <typename Futures_>
        void finalize(launch policy, Futures_&& futures)
        {
            if (policy == launch::sync)
            {
                finalize(launch::sync, PIKA_FORWARD(Futures_, futures));
            }
            else if (policy == launch::fork)
            {
                finalize(launch::fork, PIKA_FORWARD(Futures_, futures));
            }
            else
            {
                finalize(launch::async, PIKA_FORWARD(Futures_, futures));
            }
        }

        // The overload for pika::dataflow taking an executor simply forwards
        // to the corresponding executor customization point.
        template <typename Executor, typename Futures_>
        PIKA_FORCEINLINE typename std::enable_if<
            (traits::is_one_way_executor<Executor>::value ||
                traits::is_two_way_executor<Executor>::value) &&
            !has_dataflow_finalize<Executor, dataflow_frame, Func,
                Futures_>::value>::type
        finalize(Executor&& exec, Futures_&& futures)
        {
            detail::dataflow_finalization<dataflow_type> this_f_(this);

            pika::parallel::execution::post(PIKA_FORWARD(Executor, exec),
                PIKA_MOVE(this_f_), PIKA_FORWARD(Futures_, futures));
        }

        template <typename Executor, typename Futures_>
        PIKA_FORCEINLINE typename std::enable_if<
            (traits::is_one_way_executor<Executor>::value ||
                traits::is_two_way_executor<Executor>::value) &&
            has_dataflow_finalize<Executor, dataflow_frame, Func,
                Futures_>::value>::type
        finalize(Executor&& exec, Futures_&& futures)
        {
#if defined(PIKA_CUDA_VERSION)
            std::forward<Executor>(exec)
#else
            PIKA_FORWARD(Executor, exec)
#endif
                .dataflow_finalize(
                    this, PIKA_MOVE(func_), PIKA_FORWARD(Futures_, futures));
        }

    public:
        /// Check whether the current future is ready
        template <typename T>
        auto operator()(util::async_traverse_visit_tag, T&& current)
            -> decltype(async_visit_future(PIKA_FORWARD(T, current)))
        {
            return async_visit_future(PIKA_FORWARD(T, current));
        }

        /// Detach the current execution context and continue when the
        /// current future was set to be ready.
        template <typename T, typename N>
        auto operator()(util::async_traverse_detach_tag, T&& current, N&& next)
            -> decltype(async_detach_future(
                PIKA_FORWARD(T, current), PIKA_FORWARD(N, next)))
        {
            return async_detach_future(
                PIKA_FORWARD(T, current), PIKA_FORWARD(N, next));
        }

        /// Finish the dataflow when the traversal has finished
        template <typename Futures_>
        PIKA_FORCEINLINE void operator()(
            util::async_traverse_complete_tag, Futures_&& futures)
        {
            finalize(policy_, PIKA_FORWARD(Futures_, futures));
        }

    private:
        Policy policy_;
        Func func_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Policy, typename Func, typename... Ts,
        typename Frame = dataflow_frame<typename std::decay<Policy>::type,
            typename std::decay<Func>::type,
            pika::tuple<typename std::decay<Ts>::type...>>>
    typename Frame::type create_dataflow(
        Policy&& policy, Func&& func, Ts&&... ts)
    {
        // Create the data which is used to construct the dataflow_frame
        auto data = Frame::construct_from(
            PIKA_FORWARD(Policy, policy), PIKA_FORWARD(Func, func));

        // Construct the dataflow_frame and traverse
        // the arguments asynchronously
        pika::intrusive_ptr<Frame> p = util::traverse_pack_async(
            util::async_traverse_in_place_tag<Frame>{}, PIKA_MOVE(data),
            PIKA_FORWARD(Ts, ts)...);

        using traits::future_access;
        return future_access<typename Frame::type>::create(PIKA_MOVE(p));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Allocator, typename Policy, typename Func,
        typename... Ts,
        typename Frame = dataflow_frame<typename std::decay<Policy>::type,
            typename std::decay<Func>::type,
            pika::tuple<typename std::decay<Ts>::type...>>>
    typename Frame::type create_dataflow_alloc(
        Allocator const& alloc, Policy&& policy, Func&& func, Ts&&... ts)
    {
        // Create the data which is used to construct the dataflow_frame
        auto data = Frame::construct_from(
            PIKA_FORWARD(Policy, policy), PIKA_FORWARD(Func, func));

        // Construct the dataflow_frame and traverse
        // the arguments asynchronously
        pika::intrusive_ptr<Frame> p = util::traverse_pack_async_allocator(alloc,
            util::async_traverse_in_place_tag<Frame>{}, PIKA_MOVE(data),
            PIKA_FORWARD(Ts, ts)...);

        using traits::future_access;
        return future_access<typename Frame::type>::create(PIKA_MOVE(p));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <bool IsAction, typename Policy, typename Enable = void>
    struct dataflow_dispatch_impl;

    // launch
    template <typename Policy>
    struct dataflow_dispatch_impl<false, Policy,
        typename std::enable_if<traits::is_launch_policy<Policy>::value>::type>
    {
        template <typename Allocator, typename Policy_, typename F,
            typename... Ts>
        PIKA_FORCEINLINE static decltype(auto) call(
            Allocator const& alloc, Policy_&& policy, F&& f, Ts&&... ts)
        {
            return detail::create_dataflow_alloc(alloc,
                PIKA_FORWARD(Policy_, policy), PIKA_FORWARD(F, f),
                traits::acquire_future_disp()(PIKA_FORWARD(Ts, ts))...);
        }
    };

    template <typename Policy>
    struct dataflow_dispatch<Policy,
        typename std::enable_if<traits::is_launch_policy<Policy>::value>::type>
    {
        template <typename Allocator, typename F, typename... Ts>
        PIKA_FORCEINLINE static auto call(
            Allocator const& alloc, F&& f, Ts&&... ts)
            -> decltype(dataflow_dispatch_impl<false, Policy>::call(
                alloc, PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...))
        {
            return dataflow_dispatch_impl<false, Policy>::call(
                alloc, PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
        }

        template <typename Allocator, typename P, typename F, typename Id,
            typename... Ts>
        PIKA_FORCEINLINE static auto call(Allocator const& alloc, P&& p, F&& f,
            typename std::enable_if<
                traits::is_action<typename std::decay<F>::type>::value,
                Id>::type const& id,
            Ts&&... ts)
            -> decltype(dataflow_dispatch_impl<
                traits::is_action<typename std::decay<F>::type>::value,
                Policy>::call(alloc, PIKA_FORWARD(P, p), PIKA_FORWARD(F, f), id,
                PIKA_FORWARD(Ts, ts)...))
        {
            return dataflow_dispatch_impl<
                traits::is_action<typename std::decay<F>::type>::value,
                Policy>::call(alloc, PIKA_FORWARD(P, p), PIKA_FORWARD(F, f), id,
                PIKA_FORWARD(Ts, ts)...);
        }
    };

    // executors
    template <typename Executor>
    struct dataflow_dispatch<Executor,
        typename std::enable_if<traits::is_one_way_executor<Executor>::value ||
            traits::is_two_way_executor<Executor>::value>::type>
    {
        template <typename Allocator, typename Executor_, typename F,
            typename... Ts>
        PIKA_FORCEINLINE static decltype(auto) call(
            Allocator const& alloc, Executor_&& exec, F&& f, Ts&&... ts)
        {
            return detail::create_dataflow_alloc(alloc,
                PIKA_FORWARD(Executor_, exec), PIKA_FORWARD(F, f),
                traits::acquire_future_disp()(PIKA_FORWARD(Ts, ts))...);
        }
    };

    // any action, plain function, or function object
    template <typename FD>
    struct dataflow_dispatch_impl<false, FD,
        typename std::enable_if<!traits::is_launch_policy<FD>::value &&
            !(traits::is_one_way_executor<FD>::value ||
                traits::is_two_way_executor<FD>::value)>::type>
    {
        template <typename Allocator, typename F, typename... Ts,
            typename Enable = typename std::enable_if<
                !traits::is_action<typename std::decay<F>::type>::value>::type>
        PIKA_FORCEINLINE static auto call(Allocator const& alloc, F&& f,
            Ts&&... ts) -> decltype(dataflow_dispatch<launch>::call(alloc,
            launch::async, PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...))
        {
            return dataflow_dispatch<launch>::call(alloc, launch::async,
                PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
        }
    };

    template <typename FD>
    struct dataflow_dispatch<FD,
        typename std::enable_if<!traits::is_launch_policy<FD>::value &&
            !(traits::is_one_way_executor<FD>::value ||
                traits::is_two_way_executor<FD>::value)>::type>
    {
        template <typename Allocator, typename F, typename... Ts>
        PIKA_FORCEINLINE static auto call(
            Allocator const& alloc, F&& f, Ts&&... ts)
            -> decltype(dataflow_dispatch_impl<
                traits::is_action<typename std::decay<F>::type>::value,
                launch>::call(alloc, launch::async, PIKA_FORWARD(F, f),
                PIKA_FORWARD(Ts, ts)...))
        {
            return dataflow_dispatch_impl<
                traits::is_action<typename std::decay<F>::type>::value,
                launch>::call(alloc, launch::async, PIKA_FORWARD(F, f),
                PIKA_FORWARD(Ts, ts)...);
        }
    };
}}}    // namespace pika::lcos::detail
