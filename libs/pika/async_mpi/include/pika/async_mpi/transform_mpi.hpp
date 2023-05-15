//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/transform_xxx.hpp

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/async_mpi/dispatch_mpi.hpp>
#include <pika/async_mpi/mpi_polling.hpp>
#include <pika/async_mpi/trigger_mpi.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/datastructures/variant.hpp>
#include <pika/debugging/demangle_helper.hpp>
#include <pika/debugging/print.hpp>
#include <pika/execution/algorithms/detail/helpers.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution/algorithms/transfer.hpp>
#include <pika/execution_base/any_sender.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/executors/inline_scheduler.hpp>
#include <pika/executors/limiting_scheduler.hpp>
#include <pika/executors/thread_pool_scheduler.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/functional/invoke_fused.hpp>
#include <pika/mpi_base/mpi.hpp>

#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pika::mpi::experimental::detail {

    namespace pud = pika::util::detail;
    namespace exp = execution::experimental;

    // -----------------------------------------------------------------
    // route calls through an impl layer for ADL resolution
    template <typename Sender, typename F>
    struct transform_mpi_sender_impl
    {
        struct transform_mpi_sender_type;
    };

    template <typename Sender, typename F>
    using transform_mpi_sender =
        typename transform_mpi_sender_impl<Sender, F>::transform_mpi_sender_type;

    // -----------------------------------------------------------------
    // transform MPI adapter - sender type
    template <typename Sender, typename F>
    struct transform_mpi_sender_impl<Sender, F>::transform_mpi_sender_type
    {
        using is_sender = void;

        std::decay_t<Sender> sender;
        std::decay_t<F> f;
        stream_type stream;

#if defined(PIKA_HAVE_STDEXEC)
        template <typename... Ts>
        requires is_mpi_request_invocable_v<F, Ts...>
        using invoke_result_helper = exp::completion_signatures<
            exp::detail::result_type_signature_helper_t<mpi_request_invoke_result_t<F, Ts...>>>;

        using completion_signatures = exp::make_completion_signatures<std::decay_t<Sender>,
            exp::empty_env, exp::completion_signatures<exp::set_error_t(std::exception_ptr)>,
            invoke_result_helper>;
#else
        // -----------------------------------------------------------------
        // get the return tuple<type> of func (tuple<args> + MPI_Request)
        template <typename Tuple>
        struct invoke_result_helper;

        template <template <typename...> class Tuple, typename... Ts>
        struct invoke_result_helper<Tuple<Ts...>>
        {
            static_assert(is_mpi_request_invocable_v<F, Ts...>,
                "F not invocable with the value_types specified.");
            using result_type = mpi_request_invoke_result_t<F, Ts...>;
            using type =
                std::conditional_t<std::is_void<result_type>::value, Tuple<>, Tuple<result_type>>;
        };

        // -----------------------------------------------------------------
        // get pack of unique types from combined tuple and variant
        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types = pud::unique_t<pud::transform_t<
            typename exp::sender_traits<Sender>::template value_types<Tuple, Variant>,
            invoke_result_helper>>;

        template <template <typename...> class Variant>
        using error_types = pud::unique_t<
            pud::prepend_t<typename exp::sender_traits<Sender>::template error_types<Variant>,
                std::exception_ptr>>;

        static constexpr bool sends_done = false;
#endif

        // -----------------------------------------------------------------
        // operation state for a given receiver
        template <typename Receiver>
        struct operation_state
        {
            std::decay_t<Receiver> receiver;
            std::decay_t<F> f;
            stream_type stream;
            pika::spinlock mutex_;
            pika::condition_variable cond_var_;
            bool completed;
            int status;

            // -----------------------------------------------------------------
            // The mpi_receiver receives inputs from the previous sender,
            // invokes the mpi call, and sets a callback on the polling handler
            struct transform_mpi_receiver
            {
                using is_receiver = void;

                operation_state& op_state;

                template <typename Error>
                friend constexpr void
                tag_invoke(exp::set_error_t, transform_mpi_receiver&& r, Error&& error) noexcept
                {
                    exp::set_error(PIKA_MOVE(r.op_state.receiver), PIKA_FORWARD(Error, error));
                }

                friend constexpr void tag_invoke(
                    exp::set_stopped_t, transform_mpi_receiver&& r) noexcept
                {
                    exp::set_stopped(PIKA_MOVE(r.op_state.receiver));
                }

                // receive the MPI function and arguments and add a request,
                // then invoke the mpi function and set a callback to be
                // triggered when the mpi request completes
                template <typename... Ts,
                    typename = std::enable_if_t<is_mpi_request_invocable_v<F, Ts...>>>
                friend constexpr void
                tag_invoke(exp::set_value_t, transform_mpi_receiver&& r, Ts&&... ts) noexcept
                {
                    pika::detail::try_catch_exception_ptr(
                        [&]() mutable {
                            using namespace pika::debug::detail;
                            using ts_element_type = std::tuple<std::decay_t<Ts>...>;
                            using invoke_result_type = mpi_request_invoke_result_t<F, Ts...>;
                            //
                            r.op_state.ts.template emplace<ts_element_type>(
                                PIKA_FORWARD(Ts, ts)...);
                            auto& t = std::get<ts_element_type>(r.op_state.ts);
                            //
                            MPI_Request request{MPI_REQUEST_NULL};
                            // modes 0 uses the task yield_while method of callback
                            // modes 1,2 use the task resume method of callback
                            auto mode = get_completion_mode();
                            if (mode < 3)
                            {
                                pud::invoke_fused(
                                    [&](auto&... ts) mutable {
                                        PIKA_DETAIL_DP(mpi_tran,
                                            debug(str<>("mpi invoke"), dec<2>(mode),
                                                print_type<invoke_result_type>()));
                                        // execute the mpi function call, passing in the request object
                                        if constexpr (std::is_void_v<invoke_result_type>)
                                        {
                                            PIKA_INVOKE(PIKA_MOVE(r.op_state.f), ts..., &request);
                                            PIKA_ASSERT_MSG(request != MPI_REQUEST_NULL,
                                                "MPI_REQUEST_NULL returned from mpi "
                                                "invocation");
                                        }
                                        else
                                        {
                                            r.op_state.result.template emplace<invoke_result_type>(
                                                PIKA_INVOKE(
                                                    PIKA_MOVE(r.op_state.f), ts..., &request));
                                            PIKA_ASSERT_MSG(request != MPI_REQUEST_NULL,
                                                "MPI_REQUEST_NULL returned from mpi "
                                                "invocation");
                                        }
                                    },
                                    t);
                                //
                                if (mode == 0)
                                {
                                    pika::util::yield_while(
                                        [&request]() { return !detail::poll_request(request); });
                                }
                                else
                                {
                                    // don't suspend if request completed already
                                    if (!detail::poll_request(request))
                                    {
                                        // suspend is invalid except on a pika thread
                                        PIKA_ASSERT(pika::threads::detail::get_self_id());
                                        threads::detail::thread_data::scoped_thread_priority
                                            set_restore(execution::thread_priority::high);
                                        std::unique_lock l{r.op_state.mutex_};
                                        resume_request_callback(request, r.op_state);
                                        r.op_state.cond_var_.wait(
                                            l, [&]() { return r.op_state.completed; });
                                    }
                                }
                                r.op_state.ts = {};
                                r.op_state.status = MPI_SUCCESS;
                                if constexpr (!std::is_void_v<invoke_result_type>)
                                {
                                    set_value_error_helper(r.op_state.status,
                                        PIKA_MOVE(r.op_state.receiver),
                                        PIKA_MOVE(std::get<invoke_result_type>(r.op_state.result)));
                                }
                                else
                                {
                                    set_value_error_helper(
                                        r.op_state.status, PIKA_MOVE(r.op_state.receiver));
                                }
                            }
                            // modes 3,4,5,6,7,8 ....
                            else
                            {
                                PIKA_DETAIL_DP(mpi_tran,
                                    debug(str<>("mpi invoke"), dec<2>(mode),
                                        print_type<invoke_result_type>()));
                                if constexpr (std::is_void_v<invoke_result_type>)
                                {
                                    pud::invoke_fused(
                                        [&](auto&... ts) mutable {
                                            PIKA_INVOKE(PIKA_MOVE(r.op_state.f), ts..., &request);
                                            PIKA_ASSERT_MSG(request != MPI_REQUEST_NULL,
                                                "MPI_REQUEST_NULL returned from mpi "
                                                "invocation");
                                            // return type void, no value to forward to receiver
                                            set_value_request_callback_void(request, r.op_state);
                                        },
                                        t);
                                }
                                else
                                {
                                    pud::invoke_fused(
                                        [&](auto&... ts) mutable {
                                            r.op_state.result.template emplace<invoke_result_type>(
                                                PIKA_INVOKE(
                                                    PIKA_MOVE(r.op_state.f), ts..., &request));
                                            PIKA_ASSERT_MSG(request != MPI_REQUEST_NULL,
                                                "MPI_REQUEST_NULL returned from mpi "
                                                "invocation");
                                            // forward value to receiver
                                            detail::set_value_request_callback_non_void<
                                                invoke_result_type>(request, r.op_state);
                                        },
                                        t);
                                }
                            }
                        },
                        [&](std::exception_ptr ep) {
                            exp::set_error(PIKA_MOVE(r.op_state.receiver), PIKA_MOVE(ep));
                        });
                }

                friend constexpr exp::empty_env tag_invoke(
                    exp::get_env_t, transform_mpi_receiver const&) noexcept
                {
                    return {};
                }
            };

            using operation_state_type =
                exp::connect_result_t<std::decay_t<Sender>, transform_mpi_receiver>;
            operation_state_type op_state;

            template <typename Tuple>
            struct value_types_helper
            {
                using type = pud::transform_t<Tuple, std::decay>;
            };

#if defined(PIKA_HAVE_STDEXEC)
            using ts_type = pud::prepend_t<
                pud::transform_t<exp::value_types_of_t<std::decay_t<Sender>, exp::empty_env,
                                     std::tuple, pika::detail::variant>,
                    value_types_helper>,
                pika::detail::monostate>;
#else
            using ts_type = pud::prepend_t<
                pud::transform_t<typename exp::sender_traits<std::decay_t<Sender>>::
                                     template value_types<std::tuple, pika::detail::variant>,
                    value_types_helper>,
                pika::detail::monostate>;
#endif
            ts_type ts;

            // We store the return value of f in a variant. We know that
            // value_types of the transform_mpi_sender contains packs of at
            // most one element (the return value of f), so we only
            // specialize result_types_helper for zero or one value. For
            // empty packs we use pika::detail::monostate since we don't
            // need to store anything in that case.
            //
            // All in all, we:
            // - transform one-element packs to the single element, and
            //   empty packs to pika::detail::monostate
            // - add pika::detail::monostate to the pack in case it wasn't
            //   there already
            // - remove duplicates in case pika::detail::monostate has been
            //   added twice
            // - change the outer pack to a pika::detail::variant
            template <typename Tuple>
            struct result_types_helper;

            template <template <typename...> class Tuple, typename T>
            struct result_types_helper<Tuple<T>>
            {
                using type = std::decay_t<T>;
            };

            template <template <typename...> class Tuple>
            struct result_types_helper<Tuple<>>
            {
                using type = pika::detail::monostate;
            };
#if defined(PIKA_HAVE_STDEXEC)
            using result_type = pud::change_pack_t<pika::detail::variant,
                pud::unique_t<
                    pud::prepend_t<pud::transform_t<exp::value_types_of_t<transform_mpi_sender_type,
                                                        exp::empty_env, pud::pack, pud::pack>,
                                       result_types_helper>,
                        pika::detail::monostate>>>;
#else
            using result_type = pud::change_pack_t<pika::detail::variant,
                pud::unique_t<pud::prepend_t<
                    pud::transform_t<transform_mpi_sender_type::value_types<pud::pack, pud::pack>,
                        result_types_helper>,
                    pika::detail::monostate>>>;
#endif
            result_type result;

            template <typename Receiver_, typename F_, typename Sender_>
            operation_state(Receiver_&& receiver, F_&& f, Sender_&& sender, stream_type s)
              : receiver(PIKA_FORWARD(Receiver_, receiver))
              , f(PIKA_FORWARD(F_, f))
              , stream{s}
              , completed{false}
              , status{MPI_SUCCESS}
              , op_state(exp::connect(PIKA_FORWARD(Sender_, sender), transform_mpi_receiver{*this}))
            {
                PIKA_DETAIL_DP(mpi_tran,
                    debug(
                        debug::detail::str<>("operation_state"), "stream", detail::stream_name(s)));
            }

            friend constexpr auto tag_invoke(exp::start_t, operation_state& os) noexcept
            {
                return exp::start(os.op_state);
            }
        };

        template <typename Receiver>
        friend constexpr auto
        tag_invoke(exp::connect_t, transform_mpi_sender_type const& s, Receiver&& receiver)
        {
            return operation_state<Receiver>(
                PIKA_FORWARD(Receiver, receiver), s.f, s.sender, s.stream);
        }

        template <typename Receiver>
        friend constexpr auto
        tag_invoke(exp::connect_t, transform_mpi_sender_type&& s, Receiver&& receiver)
        {
            return operation_state<Receiver>(
                PIKA_FORWARD(Receiver, receiver), PIKA_MOVE(s.f), PIKA_MOVE(s.sender), s.stream);
        }
    };
}    // namespace pika::mpi::experimental::detail

namespace pika::mpi::experimental {

    namespace pud = pika::util::detail;
    namespace exp = execution::experimental;

    inline constexpr struct transform_mpi_t final
      : pika::functional::detail::tag_fallback<transform_mpi_t>
    {
    private:
        template <typename Sender, typename F,
            PIKA_CONCEPT_REQUIRES_(exp::is_sender_v<std::decay_t<Sender>>)>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(transform_mpi_t, Sender&& sender,
            F&& f, progress_mode p, stream_type s = stream_type::automatic)
        {
            using namespace pika::mpi::experimental::detail;
            PIKA_DETAIL_DP(mpi_tran,
                debug(
                    debug::detail::str<>("tag_fallback_invoke"), "stream", detail::stream_name(s)));

            using execution::thread_priority;
            using exp::make_unique_any_sender;
            using exp::schedule;
            using exp::then;
            using exp::thread_pool_scheduler;
            using exp::transfer;
            using exp::with_priority;
            using exp::with_stacksize;

            // what is the output of set_value for this transform_mpi sender
            using our_type = transform_mpi_sender<Sender, F>;
            using value_type =
                typename std::decay<exp::detail::single_result_t<typename exp::sender_traits<
                    our_type>::template value_types<pud::pack, pud::pack>>>::type;
            // this is the final sender type unique_any_sender<?>
            typename any_sender_helper<value_type>::type result;

            // does a custom mpi pool exist?
            auto mpi_exist = pool_exists();
            // get the mpi completion mode
            auto mode = get_completion_mode();
            // ----------------------------------------------------------
            // DLAF default : use yield_while (transfer to mpi pool if cannot_block)
            if (mode == 0)
            {
                if (p == progress_mode::can_block)
                {
                    auto snd1 = transform_mpi_sender<Sender, F>{
                        PIKA_FORWARD(Sender, sender), PIKA_FORWARD(F, f), s};
                    result = make_unique_any_sender(std::move(snd1));
                }
                else
                {
                    auto snd0 = PIKA_FORWARD(Sender, sender) | transfer(mpi_pool_scheduler());
                    auto snd1 = transform_mpi_sender<decltype(snd0), F>{
                        PIKA_MOVE(snd0), PIKA_FORWARD(F, f), s};
                    result = make_unique_any_sender(std::move(snd1));
                }
            }
            // ----------------------------------------------------------
            // suspend resume : run mpi inline - transfer completion if necessary
            else if (mode == 1)
            {
                auto snd0 = dispatch_mpi_sender<Sender, F>{PIKA_FORWARD(Sender, sender),
                                PIKA_FORWARD(F, f), s} |
                    exp::let_value([](MPI_Request request) -> exp::unique_any_sender<int> {
                        if (request == MPI_REQUEST_NULL)
                            return exp::just(MPI_SUCCESS);
                        else
                        {
                            return exp::just(request) | trigger_mpi();
                        }
                    });
                result = snd0;
            }
            // ----------------------------------------------------------
            // suspend resume : transfer to mpi - run completion on mpi pool
            else if (mode == 2)
            {
                auto snd0 = dispatch_mpi_sender<Sender, F>{PIKA_FORWARD(Sender, sender),
                                PIKA_FORWARD(F, f), s} |
                    exp::let_value([](MPI_Request r) -> exp::unique_any_sender<int> {
                        return (r == MPI_REQUEST_NULL) ?
                            exp::just(MPI_SUCCESS) :

                            exp::just(r) | transfer(default_pool_scheduler()) | trigger_mpi();
                    });
                result = snd0;
            }
            // ----------------------------------------------------------
            // suspend resume : transfer to mpi - transfer completion back
            else if (mode == 3)
            {
                auto snd0 = PIKA_FORWARD(Sender, sender) | transfer(mpi_pool_scheduler(false));
                auto snd1 = transform_mpi_sender<decltype(snd0), F>{std::move(snd0),
                                PIKA_FORWARD(F, f), s} |
                    transfer(default_pool_scheduler(thread_priority::high));
                result = make_unique_any_sender(std::move(snd1));
            }
            // ----------------------------------------------------------
            // polling mode : run mpi inline - HP transfer completion if necessary
            else if (mode == 4)
            {
                auto snd0 = transform_mpi_sender<Sender, F>{
                    PIKA_FORWARD(Sender, sender), PIKA_FORWARD(F, f), s};
                if (p == progress_mode::can_block)
                {
                    result = make_unique_any_sender(std::move(snd0));
                }
                else
                {
                    auto snd1 =
                        std::move(snd0) | transfer(default_pool_scheduler(thread_priority::high));
                    result = make_unique_any_sender(std::move(snd1));
                }
            }
            // ----------------------------------------------------------
            // polling mode : run mpi inline - NP transfer completion if necessary
            else if (mode == 5)
            {
                auto snd0 = transform_mpi_sender<Sender, F>{
                    PIKA_FORWARD(Sender, sender), PIKA_FORWARD(F, f), s};
                if (p == progress_mode::can_block)
                {
                    result = make_unique_any_sender(std::move(snd0));
                }
                else
                {
                    auto snd1 = std::move(snd0) | transfer(default_pool_scheduler());
                    result = make_unique_any_sender(std::move(snd1));
                }
            }
            // ----------------------------------------------------------
            // polling mode : transfer mpi - HP transfer completion
            else if (mode == 6)
            {
                auto snd0 = PIKA_FORWARD(Sender, sender) | transfer(mpi_pool_scheduler(false));
                auto snd1 = transform_mpi_sender<decltype(snd0), F>{std::move(snd0),
                                PIKA_FORWARD(F, f), s} |
                    transfer(default_pool_scheduler(thread_priority::high));
                result = make_unique_any_sender(std::move(snd1));
            }
            else if (mode == 7)
            {
                // transfer mpi to mpi pool,
                // run completion explicitly on default pool without priority
                auto snd0 = PIKA_FORWARD(Sender, sender) | transfer(mpi_pool_scheduler(false));
                auto snd1 = transform_mpi_sender<decltype(snd0), F>{std::move(snd0),
                                PIKA_FORWARD(F, f), s} |
                    transfer(default_pool_scheduler());
                result = make_unique_any_sender(std::move(snd1));
            }

            // ----------------------------
            // Modes need checking before use
            // ----------------------------
            else if (mode == 8)
            {
                // transfer mpi to mpi pool,
                // run completion on polling thread (mpi or default pool)
                auto snd0 = PIKA_FORWARD(Sender, sender) | transfer(mpi_pool_scheduler(false));
                auto snd1 =
                    transform_mpi_sender<decltype(snd0), F>{std::move(snd0), PIKA_FORWARD(F, f), s};
                result = make_unique_any_sender(std::move(snd1));
            }
            else if (mode == 9)
            {
                // transfer mpi to mpi pool
                // run completion explicitly on mpi pool as high priority
                auto snd0 = PIKA_FORWARD(Sender, sender) | transfer(mpi_pool_scheduler(false));
                auto snd1 = transform_mpi_sender<decltype(snd0), F>{std::move(snd0),
                                PIKA_FORWARD(F, f), s} |
                    transfer(with_priority(mpi_pool_scheduler(true), thread_priority::high));
                result = make_unique_any_sender(std::move(snd1));
            }
            else if (mode == 10)
            {
                // transfer mpi to mpi pool
                // run completion explicitly on default pool using high priority
                auto snd0 = PIKA_FORWARD(Sender, sender) | transfer(mpi_pool_scheduler(false));
                auto snd1 = transform_mpi_sender<decltype(snd0), F>{std::move(snd0),
                                PIKA_FORWARD(F, f), s} |
                    transfer(default_pool_scheduler(thread_priority::high));
                result = make_unique_any_sender(std::move(snd1));
            }
            else if (mode == 11)
            {
                // transfer mpi to mpi pool
                // run completion explicitly on default pool using default priority
                auto snd0 = PIKA_FORWARD(Sender, sender) | transfer(mpi_pool_scheduler(false));
                auto snd1 = transform_mpi_sender<decltype(snd0), F>{std::move(snd0),
                                PIKA_FORWARD(F, f), s} |
                    transfer(with_priority(default_pool_scheduler(), thread_priority::normal));
                result = make_unique_any_sender(std::move(snd1));
            }
            /*
 *     Temporarily disabled until bypass scheduler available
 *
                else if (mode == 11)
                {
                    // run mpi inline on current pool
                    // run completion with bypass on mpi pool
                    auto snd1 =
                        transform_mpi_sender<Sender, F>{
                            PIKA_FORWARD(Sender, sender), PIKA_FORWARD(F, f),
                            s} |
                        transfer(thread_pool_scheduler_queue_bypass{
                            &pika::resource::get_thread_pool(
                                get_pool_name())});
                    return make_unique_any_sender(std::move(snd1));
                }
                else if (mode == 12)
                {
                    // transfer mpi to mpi pool,
                    // run completion with bypass on mpi pool
                    auto snd0 = PIKA_FORWARD(Sender, sender) |
                        transfer(with_stacksize(
                            mpi_pool_scheduler(),
                            execution::thread_stacksize::nostack));
                    auto snd1 =
                        transform_mpi_sender<decltype(snd0), F>{
                            std::move(snd0), PIKA_FORWARD(F, f), s} |
                        transfer(thread_pool_scheduler_queue_bypass{
                            &pika::resource::get_thread_pool(
                                get_pool_name())});
                    return make_unique_any_sender(std::move(snd1));
                }
                else if (mode == 13)
                {
                    // transfer mpi to mpi pool
                    // run completion inline with bypass on default pool
                    // only effective if default pool is polling pool (mpi=default)
                    auto snd0 = PIKA_FORWARD(Sender, sender) |
                        transfer(with_stacksize(
                            mpi_pool_scheduler(),
                            execution::thread_stacksize::nostack));
                    auto snd1 =
                        transform_mpi_sender<decltype(snd0), F>{
                            std::move(snd0), PIKA_FORWARD(F, f), s} |
                        transfer(thread_pool_scheduler_queue_bypass{
                            &pika::resource::get_thread_pool("default")});
                    return make_unique_any_sender(std::move(snd1));
                }
*/
            else
            {
                PIKA_THROW_EXCEPTION(pika::error::bad_parameter, "transform_mpi",
                    "Unsupported transfer mode: {} (valid options are between {} and {} and "
                    "can be set with env{PIKA_MPI_COMPLETION_MODE}",
                    mode, 0, 10);
            }

            //            if constexpr (exp::detail::has_completion_scheduler_v<
            //                              exp::set_value_t, std::decay_t<Sender>>)
            //            {
            //                auto cs = exp::get_completion_scheduler<
            //                    exp::set_value_t>(PIKA_FORWARD(Sender, sender));
            //                result = make_unique_any_sender(transfer(cs, std::move(result)));
            //                throw std::runtime_error("This is never called!");
            //            }
            return result;
        }

        //
        // tag invoke overload for mpi_transform
        //
        template <typename F>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            transform_mpi_t, F&& f, progress_mode p, stream_type s = stream_type::automatic)
        {
            return exp::detail::partial_algorithm<transform_mpi_t, F, progress_mode, stream_type>{
                PIKA_FORWARD(F, f), p, s};
        }

    } transform_mpi{};
}    // namespace pika::mpi::experimental
