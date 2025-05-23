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
#include <pika/mpi_base/mpi_exception.hpp>

#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pika::dispatch_mpi_detail {
    namespace ex = execution::experimental;
    namespace mpi = pika::mpi::experimental;

    // -----------------------------------------------------------------
    // operation state for an internal receiver
    template <typename Receiver, typename F, typename Sender>
    struct operation_state
    {
        PIKA_NO_UNIQUE_ADDRESS Receiver r;
        PIKA_NO_UNIQUE_ADDRESS F f;

        // -----------------------------------------------------------------
        // The mpi_receiver receives inputs from the previous sender,
        // invokes the mpi call, and sets a callback on the polling handler
        struct receiver
        {
            PIKA_STDEXEC_RECEIVER_CONCEPT

            operation_state& op_state;

            template <typename Error>
            constexpr void set_error(Error&& error) && noexcept
            {
                auto r = std::move(*this);
                ex::set_error(std::move(r.op_state.r), std::forward<Error>(error));
            }

            friend constexpr void tag_invoke(ex::set_stopped_t, receiver r) noexcept
            {
                ex::set_stopped(std::move(r.op_state.r));
            }

            // receive the MPI function invocable + arguments and add a request,
            // then invoke the mpi function with the added request
            // if the invocation gives an error, set_error
            // otherwise return the request by passing it to set_value
            template <typename... Ts,
                typename = std::enable_if_t<mpi::detail::is_mpi_request_invocable_v<F, Ts...>>>
            constexpr void set_value(Ts&... ts) && noexcept
            {
                auto r = std::move(*this);
                pika::detail::try_catch_exception_ptr(
                    [&]() mutable {
                        using invoke_result_type =
                            mpi::detail::mpi_request_invoke_result_t<F, Ts...>;

                        PIKA_DETAIL_DP(mpi::detail::mpi_tran<5>,
                            debug(str<>("dispatch_mpi_recv"), "set_value_t"));
#ifdef PIKA_HAVE_APEX
                        apex::scoped_timer apex_post("pika::mpi::post");
#endif
                        // init a request
                        MPI_Request request{MPI_REQUEST_NULL};
                        int status = MPI_SUCCESS;
                        // execute the mpi function call, passing in the request object
                        if constexpr (std::is_void_v<invoke_result_type>)
                        {
                            PIKA_INVOKE(std::move(r.op_state.f), ts..., &request);
                        }
                        else
                        {
                            static_assert(std::is_same_v<invoke_result_type, int>);
                            status = PIKA_INVOKE(std::move(r.op_state.f), ts..., &request);
                        }
                        PIKA_DETAIL_DP(mpi::detail::mpi_tran<7>,
                            debug(str<>("dispatch_mpi_recv"), "invoke mpi", ptr(request)));

                        PIKA_ASSERT_MSG(request != MPI_REQUEST_NULL,
                            "MPI_REQUEST_NULL returned from mpi invocation");

                        if (status != MPI_SUCCESS)
                        {
                            PIKA_DETAIL_DP(mpi::detail::mpi_tran<5>,
                                debug(str<>("set_error"), "status != MPI_SUCCESS",
                                    pika::mpi::detail::error_message(status)));
                            ex::set_error(std::move(r.op_state.r),
                                std::make_exception_ptr(
                                    pika::mpi::exception(status, "dispatch mpi")));
                            return;
                        }

                        ex::set_value(std::move(r.op_state.r), request);
                    },
                    [&](std::exception_ptr ep) {
                        ex::set_error(std::move(r.op_state.r), std::move(ep));
                    });
            }

            constexpr ex::empty_env get_env() const& noexcept { return {}; }
        };

        using operation_state_type = ex::connect_result_t<Sender, receiver>;
        operation_state_type op_state;

        template <typename Receiver_, typename F_, typename Sender_>
        operation_state(Receiver_&& r, F_&& f, Sender_&& sender)
          : r(std::forward<Receiver_>(r))
          , f(std::forward<F_>(f))
          , op_state(ex::connect(std::forward<Sender_>(sender), receiver{*this}))
        {
            PIKA_DETAIL_DP(mpi::detail::mpi_tran<5>, debug(str<>("operation_state")));
        }

        void start() & noexcept { return ex::start(op_state); }
    };

    // -----------------------------------------------------------------
    // transform MPI adapter - sender type
    template <typename Sender, typename F>
    struct sender
    {
        PIKA_STDEXEC_SENDER_CONCEPT

        PIKA_NO_UNIQUE_ADDRESS Sender sender;
        PIKA_NO_UNIQUE_ADDRESS F f;

#if defined(PIKA_HAVE_STDEXEC)
        template <typename...>
        using no_value_completion = pika::execution::experimental::completion_signatures<>;
        using completion_signatures =
            pika::execution::experimental::transform_completion_signatures_of<Sender,
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

        template <typename Receiver>
        constexpr auto connect(Receiver&& receiver) const&
        {
            return operation_state<std::decay_t<Receiver>, F, Sender>(
                std::forward<Receiver>(receiver), f, sender);
        }

        template <typename Receiver>
        constexpr auto connect(Receiver&& receiver) &&
        {
            return operation_state<std::decay_t<Receiver>, F, Sender>(
                std::forward<Receiver>(receiver), std::move(f), std::move(sender));
        }
    };

}    // namespace pika::dispatch_mpi_detail

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
            return dispatch_mpi_detail::sender<std::decay_t<Sender>, std::decay_t<F>>{
                std::forward<Sender>(sender), std::forward<F>(f)};
        }

        template <typename F>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(dispatch_mpi_t, F&& f)
        {
            return pika::execution::experimental::detail::partial_algorithm<dispatch_mpi_t, F>{
                std::forward<F>(f)};
        }

    } dispatch_mpi{};

}    // namespace pika::mpi::experimental
