//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/datastructures/variant.hpp>
#include <pika/errors/try_catch_exception_ptr.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution/algorithms/then.hpp>
#include <pika/execution_base/completion_scheduler.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/detail/tag_priority_invoke.hpp>
#include <pika/functional/invoke_result.hpp>
#include <pika/iterator_support/counting_shape.hpp>
#include <pika/type_support/pack.hpp>

#include <exception>
#include <iterator>
#include <type_traits>
#include <utility>

namespace pika { namespace execution { namespace experimental {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Sender, typename Shape, typename F>
        struct bulk_sender
        {
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Shape> shape;
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types =
                typename pika::execution::experimental::sender_traits<
                    Sender>::template value_types<Tuple, Variant>;

            template <template <typename...> class Variant>
            using error_types =
                pika::util::detail::unique_t<pika::util::detail::prepend_t<
                    typename pika::execution::experimental::sender_traits<
                        Sender>::template error_types<Variant>,
                    std::exception_ptr>>;

            static constexpr bool sends_done = false;

            template <typename CPO,
                // clang-format off
                PIKA_CONCEPT_REQUIRES_(
                    pika::execution::experimental::detail::is_receiver_cpo_v<CPO> &&
                    pika::execution::experimental::detail::has_completion_scheduler_v<
                        CPO, std::decay_t<Sender>>)
                // clang-format on
                >
            friend constexpr auto tag_invoke(
                pika::execution::experimental::get_completion_scheduler_t<CPO>,
                bulk_sender const& sender)
            {
                return pika::execution::experimental::get_completion_scheduler<
                    CPO>(sender.sender);
            }

            template <typename Receiver>
            struct bulk_receiver
            {
                PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
                PIKA_NO_UNIQUE_ADDRESS std::decay_t<Shape> shape;
                PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;

                template <typename Receiver_, typename Shape_, typename F_>
                bulk_receiver(Receiver_&& receiver, Shape_&& shape, F_&& f)
                  : receiver(PIKA_FORWARD(Receiver_, receiver))
                  , shape(PIKA_FORWARD(Shape_, shape))
                  , f(PIKA_FORWARD(F_, f))
                {
                }

                template <typename Error>
                friend void tag_invoke(
                    set_error_t, bulk_receiver&& r, Error&& error) noexcept
                {
                    pika::execution::experimental::set_error(
                        PIKA_MOVE(r.receiver), PIKA_FORWARD(Error, error));
                }

                friend void tag_invoke(set_done_t, bulk_receiver&& r) noexcept
                {
                    pika::execution::experimental::set_done(
                        PIKA_MOVE(r.receiver));
                }

                template <typename... Ts>
                void set_value(Ts&&... ts)
                {
                    pika::detail::try_catch_exception_ptr(
                        [&]() {
                            for (auto const& s : shape)
                            {
                                PIKA_INVOKE(f, s, ts...);
                            }
                            pika::execution::experimental::set_value(
                                PIKA_MOVE(receiver), PIKA_FORWARD(Ts, ts)...);
                        },
                        [&](std::exception_ptr ep) {
                            pika::execution::experimental::set_error(
                                PIKA_MOVE(receiver), PIKA_MOVE(ep));
                        });
                }

                template <typename... Ts>
                friend auto tag_invoke(
                    set_value_t, bulk_receiver&& r, Ts&&... ts) noexcept
                    -> decltype(pika::execution::experimental::set_value(
                                    std::declval<std::decay_t<Receiver>&&>(),
                                    PIKA_FORWARD(Ts, ts)...),
                        void())
                {
                    // set_value is in a member function only because of a
                    // compiler bug in GCC 7. When the body of set_value is
                    // inlined here compilation fails with an internal compiler
                    // error.
                    r.set_value(PIKA_FORWARD(Ts, ts)...);
                }
            };

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, bulk_sender&& s, Receiver&& receiver)
            {
                return pika::execution::experimental::connect(PIKA_MOVE(s.sender),
                    bulk_receiver<Receiver>(PIKA_FORWARD(Receiver, receiver),
                        PIKA_MOVE(s.shape), PIKA_MOVE(s.f)));
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, bulk_sender& s, Receiver&& receiver)
            {
                return pika::execution::experimental::connect(s.sender,
                    bulk_receiver<Receiver>(
                        PIKA_FORWARD(Receiver, receiver), s.shape, s.f));
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    inline constexpr struct bulk_t final
      : pika::functional::detail::tag_priority<bulk_t>
    {
    private:
        // clang-format off
        template <typename Sender, typename Shape, typename F,
            PIKA_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                pika::execution::experimental::detail::
                    is_completion_scheduler_tag_invocable_v<
                        pika::execution::experimental::set_value_t, Sender,
                        bulk_t, Shape, F>)>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto tag_override_invoke(
            bulk_t, Sender&& sender, Shape const& shape, F&& f)
        {
            auto scheduler =
                pika::execution::experimental::get_completion_scheduler<
                    pika::execution::experimental::set_value_t>(sender);
            return pika::functional::tag_invoke(bulk_t{}, PIKA_MOVE(scheduler),
                PIKA_FORWARD(Sender, sender), shape, PIKA_FORWARD(F, f));
        }

        // clang-format off
        template <typename Sender, typename Shape, typename F,
            PIKA_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                std::is_integral<Shape>::value
            )>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            bulk_t, Sender&& sender, Shape const& shape, F&& f)
        {
            return detail::bulk_sender<Sender,
                pika::util::detail::counting_shape_type<Shape>, F>{
                PIKA_FORWARD(Sender, sender),
                pika::util::detail::make_counting_shape(shape),
                PIKA_FORWARD(F, f)};
        }

        // clang-format off
        template <typename Sender, typename Shape, typename F,
            PIKA_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                !std::is_integral<std::decay_t<Shape>>::value
            )>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            bulk_t, Sender&& sender, Shape&& shape, F&& f)
        {
            return detail::bulk_sender<Sender, Shape, F>{
                PIKA_FORWARD(Sender, sender), PIKA_FORWARD(Shape, shape),
                PIKA_FORWARD(F, f)};
        }

        template <typename Shape, typename F>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            bulk_t, Shape&& shape, F&& f)
        {
            return detail::partial_algorithm<bulk_t, Shape, F>{
                PIKA_FORWARD(Shape, shape), PIKA_FORWARD(F, f)};
        }
    } bulk{};
}}}    // namespace pika::execution::experimental
