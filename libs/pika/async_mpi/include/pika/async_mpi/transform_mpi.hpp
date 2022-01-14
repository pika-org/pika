//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/transform_xxx.hpp

#pragma once

#include <pika/local/config.hpp>
#include <pika/async_mpi/mpi_future.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/functional/traits/is_invocable.hpp>
#include <pika/mpi_base/mpi.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace pika { namespace mpi { namespace experimental {
    namespace detail {

        template <typename R, typename... Ts>
        void set_value_request_callback_helper(
            int mpi_status, R&& r, Ts&&... ts)
        {
            static_assert(sizeof...(Ts) <= 1, "Expecting at most one value");
            if (mpi_status == MPI_SUCCESS)
            {
                pika::execution::experimental::set_value(
                    PIKA_FORWARD(R, r), PIKA_FORWARD(Ts, ts)...);
            }
            else
            {
                pika::execution::experimental::set_error(PIKA_FORWARD(R, r),
                    std::make_exception_ptr(mpi_exception(mpi_status)));
            }
        }

        template <typename R, typename... Ts>
        void set_value_request_callback_void(
            MPI_Request request, R&& r, Ts&&... ts)
        {
            detail::add_request_callback(
                [r = PIKA_FORWARD(R, r),
                    keep_alive = pika::make_tuple(PIKA_FORWARD(Ts, ts)...)](
                    int status) mutable {
                    set_value_request_callback_helper(status, PIKA_MOVE(r));
                },
                request);
        }

        template <typename R, typename InvokeResult, typename... Ts>
        void set_value_request_callback_non_void(
            MPI_Request request, R&& r, InvokeResult&& res, Ts&&... ts)
        {
            detail::add_request_callback(
                [r = PIKA_FORWARD(R, r), res = PIKA_FORWARD(InvokeResult, res),
                    keep_alive = pika::make_tuple(PIKA_FORWARD(Ts, ts)...)](
                    int status) mutable {
                    set_value_request_callback_helper(
                        status, PIKA_MOVE(r), PIKA_MOVE(res));
                },
                request);
        }

        template <typename R, typename F>
        struct transform_mpi_receiver
        {
            std::decay_t<R> r;
            std::decay_t<F> f;

            template <typename R_, typename F_>
            transform_mpi_receiver(R_&& r, F_&& f)
              : r(PIKA_FORWARD(R_, r))
              , f(PIKA_FORWARD(F_, f))
            {
            }

            template <typename E>
            friend constexpr void tag_invoke(
                pika::execution::experimental::set_error_t,
                transform_mpi_receiver&& r, E&& e) noexcept
            {
                pika::execution::experimental::set_error(
                    PIKA_MOVE(r.r), PIKA_FORWARD(E, e));
            }

            friend constexpr void tag_invoke(
                pika::execution::experimental::set_done_t,
                transform_mpi_receiver&& r) noexcept
            {
                pika::execution::experimental::set_done(PIKA_MOVE(r.r));
            };

            template <typename... Ts,
                typename = std::enable_if_t<
                    pika::is_invocable_v<F, Ts..., MPI_Request*>>>
            friend constexpr void tag_invoke(
                pika::execution::experimental::set_value_t,
                transform_mpi_receiver&& r, Ts&&... ts) noexcept
            {
                pika::detail::try_catch_exception_ptr(
                    [&]() {
                        if constexpr (std::is_void_v<util::invoke_result_t<F,
                                          Ts..., MPI_Request*>>)
                        {
                            MPI_Request request;
                            PIKA_INVOKE(r.f, ts..., &request);
                            // When the return type is void, there is no value
                            // to forward to the receiver
                            set_value_request_callback_void(
                                request, PIKA_MOVE(r.r), PIKA_FORWARD(Ts, ts)...);
                        }
                        else
                        {
                            MPI_Request request;
                            // When the return type is non-void, we have to
                            // forward the value to the receiver
                            auto&& result = PIKA_INVOKE(
                                r.f, PIKA_FORWARD(Ts, ts)..., &request);
                            set_value_request_callback_non_void(request,
                                PIKA_MOVE(r.r), PIKA_MOVE(result),
                                PIKA_FORWARD(Ts, ts)...);
                        }
                    },
                    [&](std::exception_ptr ep) {
                        pika::execution::experimental::set_error(
                            PIKA_MOVE(r.r), PIKA_MOVE(ep));
                    });
            }
        };

        template <typename Sender, typename F>
        struct transform_mpi_sender
        {
            std::decay_t<Sender> s;
            std::decay_t<F> f;

            template <typename Tuple>
            struct invoke_result_helper;

            template <template <typename...> class Tuple, typename... Ts>
            struct invoke_result_helper<Tuple<Ts...>>
            {
                static_assert(pika::is_invocable_v<F, Ts..., MPI_Request*>,
                    "F not invocable with the value_types specified.");
                using result_type =
                    pika::util::invoke_result_t<F, Ts..., MPI_Request*>;
                using type =
                    std::conditional_t<std::is_void<result_type>::value,
                        Tuple<>, Tuple<result_type>>;
            };

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types =
                pika::util::detail::unique_t<pika::util::detail::transform_t<
                    typename pika::execution::experimental::sender_traits<
                        Sender>::template value_types<Tuple, Variant>,
                    invoke_result_helper>>;

            template <template <typename...> class Variant>
            using error_types =
                pika::util::detail::unique_t<pika::util::detail::prepend_t<
                    typename pika::execution::experimental::sender_traits<
                        Sender>::template error_types<Variant>,
                    std::exception_ptr>>;

            static constexpr bool sends_done = false;

            template <typename R>
            friend constexpr auto tag_invoke(
                pika::execution::experimental::connect_t,
                transform_mpi_sender& s, R&& r)
            {
                return pika::execution::experimental::connect(
                    s.s, transform_mpi_receiver<R, F>(PIKA_FORWARD(R, r), s.f));
            }

            template <typename R>
            friend constexpr auto tag_invoke(
                pika::execution::experimental::connect_t,
                transform_mpi_sender&& s, R&& r)
            {
                return pika::execution::experimental::connect(PIKA_MOVE(s.s),
                    transform_mpi_receiver<R, F>(
                        PIKA_FORWARD(R, r), PIKA_MOVE(s.f)));
            }
        };
    }    // namespace detail

    inline constexpr struct transform_mpi_t final
      : pika::functional::detail::tag_fallback<transform_mpi_t>
    {
    private:
        template <typename Sender, typename F,
            PIKA_CONCEPT_REQUIRES_(
                pika::execution::experimental::is_sender_v<Sender>)>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            transform_mpi_t, Sender&& s, F&& f)
        {
            return detail::transform_mpi_sender<Sender, F>{
                PIKA_FORWARD(Sender, s), PIKA_FORWARD(F, f)};
        }

        template <typename F>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            transform_mpi_t, F&& f)
        {
            return ::pika::execution::experimental::detail::partial_algorithm<
                transform_mpi_t, F>{PIKA_FORWARD(F, f)};
        }
    } transform_mpi{};
}}}    // namespace pika::mpi::experimental
