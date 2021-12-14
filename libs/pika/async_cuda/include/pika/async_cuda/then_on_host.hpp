//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/async_cuda/cuda_scheduler.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/type_support/pack.hpp>

#include <exception>
#include <string>
#include <type_traits>
#include <utility>

namespace pika { namespace cuda { namespace experimental {
    namespace detail {
        template <typename Receiver, typename F>
        struct then_on_host_receiver
        {
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;
            cuda_scheduler sched;

            template <typename Receiver_, typename F_>
            then_on_host_receiver(
                Receiver_&& receiver, F_&& f, cuda_scheduler sched)
              : receiver(PIKA_FORWARD(Receiver_, receiver))
              , f(PIKA_FORWARD(F_, f))
              , sched(PIKA_MOVE(sched))
            {
            }

            then_on_host_receiver(then_on_host_receiver&&) = default;
            then_on_host_receiver& operator=(then_on_host_receiver&&) = default;
            then_on_host_receiver(then_on_host_receiver const&) = delete;
            then_on_host_receiver& operator=(
                then_on_host_receiver const&) = delete;

            template <typename Error>
            friend void tag_invoke(pika::execution::experimental::set_error_t,
                then_on_host_receiver&& r, Error&& error) noexcept
            {
                pika::execution::experimental::set_error(
                    PIKA_MOVE(r.receiver), PIKA_FORWARD(Error, error));
            }

            friend void tag_invoke(pika::execution::experimental::set_done_t,
                then_on_host_receiver&& r) noexcept
            {
                pika::execution::experimental::set_done(PIKA_MOVE(r.receiver));
            }

            template <typename... Ts>
            void set_value(Ts&&... ts) noexcept
            {
                pika::detail::try_catch_exception_ptr(
                    [&]() {
                        if constexpr (std::is_void_v<pika::util::
                                              invoke_result_t<F, Ts...>>)
                        {
                        // Certain versions of GCC with optimizations fail on
                        // the move with an internal compiler error.
#if defined(PIKA_GCC_VERSION) && (PIKA_GCC_VERSION < 100000)
                            PIKA_INVOKE(std::move(f), PIKA_FORWARD(Ts, ts)...);
#else
                            PIKA_INVOKE(PIKA_MOVE(f), PIKA_FORWARD(Ts, ts)...);
#endif
                            pika::execution::experimental::set_value(
                                PIKA_MOVE(receiver));
                        }
                        else
                        {
                        // Certain versions of GCC with optimizations fail on
                        // the move with an internal compiler error.
#if defined(PIKA_GCC_VERSION) && (PIKA_GCC_VERSION < 100000)
                            auto&& result = PIKA_INVOKE(
                                std::move(f), PIKA_FORWARD(Ts, ts)...);
#else
                            auto&& result = PIKA_INVOKE(
                                PIKA_MOVE(f), PIKA_FORWARD(Ts, ts)...);
#endif
                            pika::execution::experimental::set_value(
                                PIKA_MOVE(receiver), PIKA_MOVE(result));
                        }
                    },
                    [&](std::exception_ptr ep) {
                        pika::execution::experimental::set_error(
                            PIKA_MOVE(receiver), PIKA_MOVE(ep));
                    });
            }
        };

        template <typename S, typename F>
        struct then_on_host_sender
        {
            std::decay_t<S> s;
            std::decay_t<F> f;
            cuda_scheduler sched;

            template <typename S_, typename F_>
            then_on_host_sender(S_&& s, F_&& f, cuda_scheduler sched)
              : s(PIKA_FORWARD(S_, s))
              , f(PIKA_FORWARD(F_, f))
              , sched(PIKA_MOVE(sched))
            {
            }

            then_on_host_sender(then_on_host_sender&&) = default;
            then_on_host_sender& operator=(then_on_host_sender&&) = default;
            then_on_host_sender(then_on_host_sender const&) = default;
            then_on_host_sender& operator=(
                then_on_host_sender const&) = default;

            template <typename Tuple>
            struct invoke_result_helper;

            template <template <typename...> class Tuple, typename... Ts>
            struct invoke_result_helper<Tuple<Ts...>>
            {
                using result_type =
                    typename pika::util::invoke_result<F, Ts...>::type;
                using type =
                    typename std::conditional<std::is_void<result_type>::value,
                        Tuple<>, Tuple<result_type>>::type;
            };

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types =
                pika::util::detail::unique_t<pika::util::detail::transform_t<
                    typename pika::execution::experimental::sender_traits<
                        S>::template value_types<Tuple, Variant>,
                    invoke_result_helper>>;

            template <template <typename...> class Variant>
            using error_types =
                pika::util::detail::unique_t<pika::util::detail::prepend_t<
                    typename pika::execution::experimental::sender_traits<
                        S>::template error_types<Variant>,
                    std::exception_ptr>>;

            static constexpr bool sends_done = false;

            template <typename Receiver>
            friend auto tag_invoke(pika::execution::experimental::connect_t,
                then_on_host_sender&& s, Receiver&& receiver)
            {
                return pika::execution::experimental::connect(PIKA_MOVE(s.s),
                    then_on_host_receiver<Receiver, F>{
                        PIKA_FORWARD(Receiver, receiver), PIKA_MOVE(s.f),
                        PIKA_MOVE(s.sched)});
            }

            template <typename Receiver>
            friend auto tag_invoke(pika::execution::experimental::connect_t,
                then_on_host_sender& s, Receiver&& receiver)
            {
                return pika::execution::experimental::connect(s.s,
                    then_on_host_receiver<Receiver, F>{
                        PIKA_FORWARD(Receiver, receiver), s.f, s.sched});
            }

            friend cuda_scheduler tag_invoke(
                pika::execution::experimental::get_completion_scheduler_t<
                    pika::execution::experimental::set_value_t>,
                then_on_host_sender const& s)
            {
                return s.sched;
            }
        };

        // This should be a hidden friend in then_on_host_receiver. However,
        // nvcc does not know how to compile it with some argument types
        // ("error: no instance of overloaded function std::forward matches the
        // argument list").
        template <typename Receiver, typename F, typename... Ts>
        void tag_invoke(pika::execution::experimental::set_value_t,
            then_on_host_receiver<Receiver, F>&& receiver, Ts&&... ts)
        {
            receiver.set_value(PIKA_FORWARD(Ts, ts)...);
        }
    }    // namespace detail

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
        template <typename S, typename F>
        friend PIKA_FORCEINLINE auto tag_fallback_invoke(
            then_on_host_t, S&& s, F&& f)
        {
            auto completion_sched =
                pika::execution::experimental::get_completion_scheduler<
                    pika::execution::experimental::set_value_t>(s);
            static_assert(
                std::is_same_v<std::decay_t<decltype(completion_sched)>,
                    cuda_scheduler>,
                "then_on_host can only be used with senders whose "
                "completion scheduler is cuda_scheduler");

            return detail::then_on_host_sender<S, F>{PIKA_FORWARD(S, s),
                PIKA_FORWARD(F, f), std::move(completion_sched)};
        }

        template <typename F>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            then_on_host_t, F&& f)
        {
            return pika::execution::experimental::detail::partial_algorithm<
                then_on_host_t, F>{PIKA_FORWARD(F, f)};
        }
    } then_on_host{};
}}}    // namespace pika::cuda::experimental
