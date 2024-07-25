//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/async_cuda/cuda_scheduler.hpp>
#include <pika/errors/try_catch_exception_ptr.hpp>
#include <pika/execution/algorithms/detail/helpers.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/type_support/pack.hpp>

#include <exception>
#include <string>
#include <type_traits>
#include <utility>

namespace pika::cuda::experimental {
    namespace then_on_host_detail {
        template <typename Receiver, typename F>
        struct then_on_host_receiver_impl
        {
            struct then_on_host_receiver_type;
        };

        template <typename Receiver, typename F>
        using then_on_host_receiver =
            typename then_on_host_receiver_impl<Receiver, F>::then_on_host_receiver_type;

        template <typename Receiver, typename F>
        struct then_on_host_receiver_impl<Receiver, F>::then_on_host_receiver_type
        {
            PIKA_STDEXEC_RECEIVER_CONCEPT

            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;
            cuda_scheduler sched;

            template <typename Receiver_, typename F_>
            then_on_host_receiver_type(Receiver_&& receiver, F_&& f, cuda_scheduler sched)
              : receiver(PIKA_FORWARD(Receiver_, receiver))
              , f(PIKA_FORWARD(F_, f))
              , sched(PIKA_MOVE(sched))
            {
            }

            then_on_host_receiver_type(then_on_host_receiver_type&&) = default;
            then_on_host_receiver_type& operator=(then_on_host_receiver_type&&) = default;
            then_on_host_receiver_type(then_on_host_receiver_type const&) = delete;
            then_on_host_receiver_type& operator=(then_on_host_receiver_type const&) = delete;

            template <typename Error>
            friend void tag_invoke(pika::execution::experimental::set_error_t,
                then_on_host_receiver_type&& r, Error&& error) noexcept
            {
                pika::execution::experimental::set_error(
                    PIKA_MOVE(r.receiver), PIKA_FORWARD(Error, error));
            }

            friend void tag_invoke(pika::execution::experimental::set_stopped_t,
                then_on_host_receiver_type&& r) noexcept
            {
                pika::execution::experimental::set_stopped(PIKA_MOVE(r.receiver));
            }

            template <typename... Ts>
            friend void tag_invoke(pika::execution::experimental::set_value_t,
                then_on_host_receiver_type&& r, Ts&&... ts) noexcept
            {
                pika::detail::try_catch_exception_ptr(
                    [&]() {
                        if constexpr (std::is_void_v<std::invoke_result_t<F, Ts...>>)
                        {
                        // Certain versions of GCC with optimizations fail on
                        // the move with an internal compiler error.
#if defined(PIKA_GCC_VERSION) && (PIKA_GCC_VERSION < 100000)
                            PIKA_INVOKE(std::move(r.f), PIKA_FORWARD(Ts, ts)...);
#else
                            PIKA_INVOKE(PIKA_MOVE(r.f), PIKA_FORWARD(Ts, ts)...);
#endif
                            pika::execution::experimental::set_value(PIKA_MOVE(r.receiver));
                        }
                        else
                        {
                        // Certain versions of GCC with optimizations fail on
                        // the move with an internal compiler error.
#if defined(PIKA_GCC_VERSION) && (PIKA_GCC_VERSION < 100000)
                            pika::execution::experimental::set_value(PIKA_MOVE(r.receiver),
                                PIKA_INVOKE(std::move(r.f), PIKA_FORWARD(Ts, ts)...));
#else
                            pika::execution::experimental::set_value(PIKA_MOVE(r.receiver),
                                PIKA_INVOKE(PIKA_MOVE(r.f), PIKA_FORWARD(Ts, ts)...));
#endif
                        }
                    },
                    [&](std::exception_ptr ep) {
                        pika::execution::experimental::set_error(
                            PIKA_MOVE(r.receiver), PIKA_MOVE(ep));
                    });
            }

            friend constexpr pika::execution::experimental::empty_env tag_invoke(
                pika::execution::experimental::get_env_t,
                then_on_host_receiver_type const&) noexcept
            {
                return {};
            }
        };

        template <typename Sender, typename F>
        struct then_on_host_sender_impl
        {
            struct then_on_host_sender_type;
        };

        template <typename Sender, typename F>
        using then_on_host_sender =
            typename then_on_host_sender_impl<Sender, F>::then_on_host_sender_type;

        template <typename Sender, typename F>
        struct then_on_host_sender_impl<Sender, F>::then_on_host_sender_type
        {
            PIKA_STDEXEC_SENDER_CONCEPT

            std::decay_t<Sender> sender;
            std::decay_t<F> f;
            cuda_scheduler sched;

            template <typename Sender_, typename F_>
            then_on_host_sender_type(Sender_&& sender, F_&& f, cuda_scheduler sched)
              : sender(PIKA_FORWARD(Sender_, sender))
              , f(PIKA_FORWARD(F_, f))
              , sched(PIKA_MOVE(sched))
            {
            }

            then_on_host_sender_type(then_on_host_sender_type&&) = default;
            then_on_host_sender_type& operator=(then_on_host_sender_type&&) = default;
            then_on_host_sender_type(then_on_host_sender_type const&) = default;
            then_on_host_sender_type& operator=(then_on_host_sender_type const&) = default;

#if defined(PIKA_HAVE_STDEXEC)
            template <typename... Ts>
                requires std::is_invocable_v<F, Ts...>
            using invoke_result_helper =
                pika::execution::experimental::completion_signatures<pika::execution::experimental::
                        detail::result_type_signature_helper_t<std::invoke_result_t<F, Ts...>>>;

            using completion_signatures =
                pika::execution::experimental::transform_completion_signatures_of<Sender,
                    pika::execution::experimental::empty_env,
                    pika::execution::experimental::completion_signatures<
                        pika::execution::experimental::set_error_t(std::exception_ptr)>,
                    invoke_result_helper>;
#else
            template <typename Tuple>
            struct invoke_result_helper;

            template <template <typename...> class Tuple, typename... Ts>
            struct invoke_result_helper<Tuple<Ts...>>
            {
                using result_type = std::invoke_result_t<F, Ts...>;
                using type = typename std::conditional<std::is_void<result_type>::value, Tuple<>,
                    Tuple<result_type>>::type;
            };

            template <template <typename...> class Tuple, template <typename...> class Variant>
            using value_types = pika::util::detail::unique_t<pika::util::detail::transform_t<
                typename pika::execution::experimental::sender_traits<Sender>::template value_types<
                    Tuple, Variant>,
                invoke_result_helper>>;

            template <template <typename...> class Variant>
            using error_types = pika::util::detail::unique_t<
                pika::util::detail::prepend_t<typename pika::execution::experimental::sender_traits<
                                                  Sender>::template error_types<Variant>,
                    std::exception_ptr>>;

            static constexpr bool sends_done = false;
#endif

            template <typename Receiver>
            friend auto tag_invoke(pika::execution::experimental::connect_t,
                then_on_host_sender_type&& s, Receiver&& receiver)
            {
                return pika::execution::experimental::connect(PIKA_MOVE(s.sender),
                    then_on_host_receiver<Receiver, F>{
                        PIKA_FORWARD(Receiver, receiver), PIKA_MOVE(s.f), PIKA_MOVE(s.sched)});
            }

            template <typename Receiver>
            friend auto tag_invoke(pika::execution::experimental::connect_t,
                then_on_host_sender_type const& s, Receiver&& receiver)
            {
                return pika::execution::experimental::connect(s.sender,
                    then_on_host_receiver<Receiver, F>{
                        PIKA_FORWARD(Receiver, receiver), s.f, s.sched});
            }

            friend auto tag_invoke(pika::execution::experimental::get_env_t,
                then_on_host_sender_type const& s) noexcept
            {
                return pika::execution::experimental::get_env(s.sender);
            }
        };
    }    // namespace then_on_host_detail

    // NOTE: This is not a customization of pika::execution::experimental::then.
    // It retains the cuda_scheduler execution context from the predecessor
    // sender, but does not run the continuation on a CUDA device. Instead, it
    // runs the continuation in the polling thread used by the cuda_scheduler on
    // the CPU. The continuation is run only after synchronizing all previous
    // events scheduled on the cuda_scheduler. Blocking in the callable given to
    // then_on_host blocks other work scheduled on cuda_scheduler from
    // completing. Heavier work should be transferred to a host scheduler as
    // soon as possible.
    inline constexpr struct then_on_host_t final
      : pika::functional::detail::tag_fallback<then_on_host_t>
    {
    private:
        template <typename Sender, typename F>
        friend PIKA_FORCEINLINE auto tag_fallback_invoke(then_on_host_t, Sender&& sender, F&& f)
        {
            auto completion_sched = pika::execution::experimental::get_completion_scheduler<
                pika::execution::experimental::set_value_t>(
                pika::execution::experimental::get_env(sender));
            static_assert(std::is_same_v<std::decay_t<decltype(completion_sched)>, cuda_scheduler>,
                "then_on_host can only be used with senders whose completion scheduler is "
                "cuda_scheduler");

            return then_on_host_detail::then_on_host_sender<Sender, F>{
                PIKA_FORWARD(Sender, sender), PIKA_FORWARD(F, f), std::move(completion_sched)};
        }

        template <typename F>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(then_on_host_t, F&& f)
        {
            return pika::execution::experimental::detail::partial_algorithm<then_on_host_t, F>{
                PIKA_FORWARD(F, f)};
        }
    } then_on_host{};
}    // namespace pika::cuda::experimental
