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
#include <pika/execution/algorithms/transfer.hpp>
#include <pika/execution_base/any_sender.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
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
        using is_sender = void;

        std::decay_t<Sender> sender;
        std::decay_t<F> f;
        stream_type stream;

        // -----------------------------------------------------------------
        // completion signatures
        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types = Variant<Tuple<MPI_Request>>;

        template <template <typename...> class Variant>
        using error_types = pud::unique_t<
            pud::prepend_t<typename exp::sender_traits<Sender>::template error_types<Variant>,
                std::exception_ptr>>;

        static constexpr bool sends_done = false;

        // -----------------------------------------------------------------
        // operation state for a internal receiver
        template <typename Receiver>
        struct operation_state
        {
            std::decay_t<Receiver> receiver;
            std::decay_t<F> f;
            stream_type stream_;

            // -----------------------------------------------------------------
            // The mpi_receiver receives inputs from the previous sender,
            // invokes the mpi call, and sets a callback on the polling handler
            struct dispatch_mpi_receiver
            {
                using is_receiver = void;

                operation_state& op_state;

                template <typename Error>
                friend constexpr void
                tag_invoke(exp::set_error_t, dispatch_mpi_receiver&& r, Error&& error) noexcept
                {
                    exp::set_error(PIKA_MOVE(r.op_state.receiver), PIKA_FORWARD(Error, error));
                }

                friend constexpr void tag_invoke(
                    exp::set_stopped_t, dispatch_mpi_receiver&& r) noexcept
                {
                    exp::set_stopped(PIKA_MOVE(r.op_state.receiver));
                }

                // receive the MPI function invocable + arguments and add a request,
                // then invoke the mpi function with the added request
                // if the invocation gives an error, set_error
                // otherwise return the request by passing it to set_value
                template <typename... Ts,
                    typename = std::enable_if_t<is_mpi_request_invocable_v<F, Ts...>>>
                friend constexpr void
                tag_invoke(exp::set_value_t, dispatch_mpi_receiver&& r, Ts&&... ts) noexcept
                {
                    pika::detail::try_catch_exception_ptr(
                        [&]() mutable {
                            using namespace pika::debug::detail;
                            using ts_element_type = std::tuple<std::decay_t<Ts>...>;
                            using invoke_result_type = mpi_request_invoke_result_t<F, Ts...>;
                            using namespace pika::debug::detail;
                            PIKA_DETAIL_DP(mpi_tran<5>,
                                debug(str<>("dispatch_mpi_recv"), "set_value_t", "stream",
                                    detail::stream_name(r.op_state.stream_)));

                            // move/copy the received params into our local opstate
                            r.op_state.ts.template emplace<ts_element_type>(
                                PIKA_FORWARD(Ts, ts)...);
                            // and get a reference to the tuple of local params
                            auto& t = std::get<ts_element_type>(r.op_state.ts);
                            // init a request
                            MPI_Request request;
                            int status = MPI_SUCCESS;
                            // invoke the function, with the contents of the param tuple as args
                            pud::invoke_fused(
                                [&](auto&... ts) mutable {
                                    // execute the mpi function call, passing in the request object
                                    if constexpr (std::is_void_v<invoke_result_type>)
                                    {
                                        PIKA_INVOKE(PIKA_MOVE(r.op_state.f), ts..., &request);
                                    }
                                    else
                                    {
                                        status =
                                            PIKA_INVOKE(PIKA_MOVE(r.op_state.f), ts..., &request);
                                    }
                                    PIKA_DETAIL_DP(mpi_tran<7>,
                                        debug(str<>("dispatch_mpi_recv"), "invoke mpi",
                                            detail::stream_name(r.op_state.stream_), request));

                                    PIKA_ASSERT_MSG(request != MPI_REQUEST_NULL,
                                        "MPI_REQUEST_NULL returned from mpi invocation");
                                },
                                t);
                            // function called, Ts... can now be released (if refs hold lifetime)
                            r.op_state.ts = {};
                            if (poll_request(request))
                            {
                                PIKA_DETAIL_DP(mpi_tran<7>,
                                    debug(str<>("dispatch_mpi_recv"), "eager poll ok",
                                        detail::stream_name(r.op_state.stream_), request));
                                // calls set_value(request), or set_error(mpi_exception(status))
                                set_value_error_helper(
                                    status, PIKA_MOVE(r.op_state.receiver), MPI_REQUEST_NULL);
                            }
                            else
                            {
                                set_value_error_helper(
                                    status, PIKA_MOVE(r.op_state.receiver), request);
                            }
                        },
                        [&](std::exception_ptr ep) {
                            exp::set_error(PIKA_MOVE(r.op_state.receiver), PIKA_MOVE(ep));
                        });
                }

                friend constexpr exp::empty_env tag_invoke(
                    exp::get_env_t, dispatch_mpi_receiver const&) noexcept
                {
                    return {};
                }
            };

            using operation_state_type =
                exp::connect_result_t<std::decay_t<Sender>, dispatch_mpi_receiver>;
            operation_state_type op_state;

            template <typename Tuple>
            struct value_types_helper
            {
                using type = pud::transform_t<Tuple, std::decay>;
            };

#if defined(PIKA_HAVE_STDEXEC)
            using ts_type = pika::util::detail::prepend_t<
                pika::util::detail::transform_t<
                    execution::experimental::value_types_of_t<std::decay_t<Sender>,
                        execution::experimental::empty_env, std::tuple, pika::detail::variant>,
                    value_types_helper>,
                pika::detail::monostate>;
#else
            using ts_type = pika::util::detail::prepend_t<
                pika::util::detail::transform_t<
                    typename execution::experimental::sender_traits<std::decay_t<Sender>>::
                        template value_types<std::tuple, pika::detail::variant>,
                    value_types_helper>,
                pika::detail::monostate>;
#endif
            ts_type ts;

            // we don't bother storing the result, but the correct type is ...
            using result_type = pika::detail::variant<std::tuple<MPI_Request>>;

            template <typename Receiver_, typename F_, typename Sender_>
            operation_state(Receiver_&& receiver, F_&& f, Sender_&& sender, stream_type s)
              : receiver(PIKA_FORWARD(Receiver_, receiver))
              , f(PIKA_FORWARD(F_, f))
              , stream_{s}
              , op_state(exp::connect(PIKA_FORWARD(Sender_, sender), dispatch_mpi_receiver{*this}))
            {
                using namespace pika::debug::detail;
                PIKA_DETAIL_DP(
                    mpi_tran<5>, debug(str<>("operation_state"), "stream", detail::stream_name(s)));
            }

            friend constexpr auto tag_invoke(exp::start_t, operation_state& os) noexcept
            {
                return exp::start(os.op_state);
            }
        };

        template <typename Receiver>
        friend constexpr auto
        tag_invoke(exp::connect_t, dispatch_mpi_sender_type const& s, Receiver&& receiver)
        {
            return operation_state<Receiver>(
                PIKA_FORWARD(Receiver, receiver), s.f, s.sender, s.stream);
        }

        template <typename Receiver>
        friend constexpr auto
        tag_invoke(exp::connect_t, dispatch_mpi_sender_type&& s, Receiver&& receiver)
        {
            return operation_state<Receiver>(
                PIKA_FORWARD(Receiver, receiver), PIKA_MOVE(s.f), PIKA_MOVE(s.sender), s.stream);
        }
    };

}    // namespace pika::mpi::experimental::detail

namespace pika::mpi::experimental {

    namespace exp = pika::execution::experimental;

    inline constexpr struct dispatch_mpi_t final
      : pika::functional::detail::tag_fallback<dispatch_mpi_t>
    {
    private:
        template <typename Sender, typename F,
            PIKA_CONCEPT_REQUIRES_(exp::is_sender_v<std::decay_t<Sender>>)>
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(dispatch_mpi_t, Sender&& sender, F&& f, stream_type s)
        {
            auto snd1 = detail::dispatch_mpi_sender<Sender, F>{
                PIKA_FORWARD(Sender, sender), PIKA_FORWARD(F, f), s};
            return exp::make_unique_any_sender(std::move(snd1));
        }

        //
        // tag invoke overload for mpi_dispatch
        //
        template <typename F>
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(dispatch_mpi_t, F&& f, stream_type s)
        {
            return exp::detail::partial_algorithm<dispatch_mpi_t, F>{PIKA_FORWARD(F, f), s};
        }

    } dispatch_mpi{};

}    // namespace pika::mpi::experimental
