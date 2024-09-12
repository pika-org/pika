//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/transform_xxx.hpp

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/async_mpi/mpi_helpers.hpp>
#include <pika/async_mpi/mpi_polling.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/datastructures/variant.hpp>
#include <pika/debugging/demangle_helper.hpp>
#include <pika/debugging/print.hpp>
#include <pika/execution/algorithms/detail/helpers.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution_base/any_sender.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/executors/thread_pool_scheduler.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/mpi_base/mpi.hpp>

#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pika::mpi::experimental::detail {
    namespace ex = execution::experimental;

    // -----------------------------------------------------------------
    // route calls through an impl layer for ADL isolation
    template <typename Sender, typename F>
    struct dispatch_mpi_sender_impl
    {
        struct dispatch_mpi_sender_type;
    };

    template <typename Sender, typename F>
    using dispatch_mpi_sender =
        typename dispatch_mpi_sender_impl<Sender, F>::dispatch_mpi_sender_type;

    // -----------------------------------------------------------------
    // transform MPI adapter - sender type
    template <typename Sender, typename F>
    struct dispatch_mpi_sender_impl<Sender, F>::dispatch_mpi_sender_type
    {
        PIKA_STDEXEC_SENDER_CONCEPT

        std::decay_t<Sender> sender;
        std::decay_t<F> f;

#if defined(PIKA_HAVE_STDEXEC)
        template <typename...>
        using no_value_completion = pika::execution::experimental::completion_signatures<>;
        using completion_signatures =
            pika::execution::experimental::transform_completion_signatures_of<std::decay_t<Sender>,
                pika::execution::experimental::empty_env,
                pika::execution::experimental::completion_signatures<
                    pika::execution::experimental::set_value_t(MPI_Request),
                    pika::execution::experimental::set_error_t(std::exception_ptr)>,
                no_value_completion>;
#else
        // -----------------------------------------------------------------
        // completion signatures
        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types = Variant<Tuple<MPI_Request>>;

        template <template <typename...> class Variant>
        using error_types = util::detail::unique_t<util::detail::prepend_t<
            typename ex::sender_traits<Sender>::template error_types<Variant>, std::exception_ptr>>;

#endif

        static constexpr bool sends_done = false;

        // -----------------------------------------------------------------
        // operation state for an internal receiver
        template <typename Receiver>
        struct operation_state
        {
            std::decay_t<Receiver> receiver;
            std::decay_t<F> f;

            // -----------------------------------------------------------------
            // The mpi_receiver receives inputs from the previous sender,
            // invokes the mpi call, and sets a callback on the polling handler
            struct dispatch_mpi_receiver
            {
                PIKA_STDEXEC_RECEIVER_CONCEPT

                operation_state& op_state;

                template <typename Error>
                friend constexpr void
                tag_invoke(ex::set_error_t, dispatch_mpi_receiver r, Error&& error) noexcept
                {
                    ex::set_error(PIKA_MOVE(r.op_state.receiver), PIKA_FORWARD(Error, error));
                }

                friend constexpr void tag_invoke(
                    ex::set_stopped_t, dispatch_mpi_receiver r) noexcept
                {
                    ex::set_stopped(PIKA_MOVE(r.op_state.receiver));
                }

                // receive the MPI function invocable + arguments and add a request,
                // then invoke the mpi function with the added request
                // if the invocation gives an error, set_error
                // otherwise return the request by passing it to set_value
                template <typename... Ts,
                    typename = std::enable_if_t<is_mpi_request_invocable_v<F, Ts...>>>
                friend constexpr void
                tag_invoke(ex::set_value_t, dispatch_mpi_receiver r, Ts&&... ts) noexcept
                {
                    pika::detail::try_catch_exception_ptr(
                        [&]() mutable {
                            using invoke_result_type = mpi_request_invoke_result_t<F, Ts...>;

                            PIKA_DETAIL_DP(
                                mpi_tran<5>, debug(str<>("dispatch_mpi_recv"), "set_value_t"));
#ifdef PIKA_HAVE_APEX
                            apex::scoped_timer apex_post("pika::mpi::post");
#endif
                            // init a request
                            MPI_Request request{MPI_REQUEST_NULL};
                            int status = MPI_SUCCESS;
                            // execute the mpi function call, passing in the request object
                            if constexpr (std::is_void_v<invoke_result_type>)
                            {
                                PIKA_INVOKE(PIKA_MOVE(r.op_state.f), ts..., &request);
                            }
                            else
                            {
                                static_assert(std::is_same_v<invoke_result_type, int>);
                                status = PIKA_INVOKE(PIKA_MOVE(r.op_state.f), ts..., &request);
                            }
                            PIKA_DETAIL_DP(mpi_tran<7>,
                                debug(str<>("dispatch_mpi_recv"), "invoke mpi", ptr(request)));

                            PIKA_ASSERT_MSG(request != MPI_REQUEST_NULL,
                                "MPI_REQUEST_NULL returned from mpi invocation");

                            if (status != MPI_SUCCESS)
                            {
                                PIKA_DETAIL_DP(mpi_tran<5>,
                                    debug(str<>("set_error"), "status != MPI_SUCCESS",
                                        detail::error_message(status)));
                                ex::set_error(PIKA_MOVE(r.op_state.receiver),
                                    std::make_exception_ptr(mpi_exception(status)));
                                return;
                            }
                            // early poll just in case the request completed immediately
                            if (poll_request(request))
                            {
#ifdef PIKA_HAVE_APEX
                                apex::scoped_timer apex_invoke("pika::mpi::trigger");
#endif
                                PIKA_DETAIL_DP(mpi_tran<7>,
                                    debug(
                                        str<>("dispatch_mpi_recv"), "eager poll ok", ptr(request)));
                                ex::set_value(PIKA_MOVE(r.op_state.receiver), MPI_REQUEST_NULL);
                            }
                            else { ex::set_value(PIKA_MOVE(r.op_state.receiver), request); }
                        },
                        [&](std::exception_ptr ep) {
                            ex::set_error(PIKA_MOVE(r.op_state.receiver), PIKA_MOVE(ep));
                        });
                }

                friend constexpr ex::empty_env tag_invoke(
                    ex::get_env_t, dispatch_mpi_receiver const&) noexcept
                {
                    return {};
                }
            };

            using operation_state_type =
                ex::connect_result_t<std::decay_t<Sender>, dispatch_mpi_receiver>;
            operation_state_type op_state;

            template <typename Tuple>
            struct value_types_helper
            {
                using type = util::detail::transform_t<Tuple, std::decay>;
            };

            template <typename Receiver_, typename F_, typename Sender_>
            operation_state(Receiver_&& receiver, F_&& f, Sender_&& sender)
              : receiver(PIKA_FORWARD(Receiver_, receiver))
              , f(PIKA_FORWARD(F_, f))
              , op_state(ex::connect(PIKA_FORWARD(Sender_, sender), dispatch_mpi_receiver{*this}))
            {
                PIKA_DETAIL_DP(mpi_tran<5>, debug(str<>("operation_state")));
            }

            friend constexpr auto tag_invoke(ex::start_t, operation_state& os) noexcept
            {
                return ex::start(os.op_state);
            }
        };

        template <typename Receiver>
        friend constexpr auto
        tag_invoke(ex::connect_t, dispatch_mpi_sender_type const& s, Receiver&& receiver)
        {
            return operation_state<Receiver>(PIKA_FORWARD(Receiver, receiver), s.f, s.sender);
        }

        template <typename Receiver>
        friend constexpr auto
        tag_invoke(ex::connect_t, dispatch_mpi_sender_type&& s, Receiver&& receiver)
        {
            return operation_state<Receiver>(
                PIKA_FORWARD(Receiver, receiver), PIKA_MOVE(s.f), PIKA_MOVE(s.sender));
        }
    };

}    // namespace pika::mpi::experimental::detail

namespace pika::mpi::experimental {
    inline constexpr struct dispatch_mpi_t final
      : pika::functional::detail::tag_fallback<dispatch_mpi_t>
    {
    private:
        template <typename Sender, typename F,
            PIKA_CONCEPT_REQUIRES_(
                pika::execution::experimental::is_sender_v<std::decay_t<Sender>>)>
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(dispatch_mpi_t, Sender&& sender, F&& f)
        {
            auto snd1 = detail::dispatch_mpi_sender<Sender, F>{
                PIKA_FORWARD(Sender, sender), PIKA_FORWARD(F, f)};
            return pika::execution::experimental::make_unique_any_sender(std::move(snd1));
        }

        template <typename F>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(dispatch_mpi_t, F&& f)
        {
            return pika::execution::experimental::detail::partial_algorithm<dispatch_mpi_t, F>{
                PIKA_FORWARD(F, f)};
        }

    } dispatch_mpi{};

}    // namespace pika::mpi::experimental
