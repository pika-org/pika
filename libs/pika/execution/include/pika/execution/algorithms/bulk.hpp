//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_STDEXEC)
# include <pika/execution_base/stdexec_forward.hpp>
#else
# include <pika/concepts/concepts.hpp>
# include <pika/datastructures/variant.hpp>
# include <pika/errors/try_catch_exception_ptr.hpp>
# include <pika/execution/algorithms/detail/partial_algorithm.hpp>
# include <pika/execution/algorithms/then.hpp>
# include <pika/execution_base/completion_scheduler.hpp>
# include <pika/execution_base/receiver.hpp>
# include <pika/execution_base/sender.hpp>
# include <pika/functional/detail/tag_priority_invoke.hpp>
# include <pika/iterator_support/counting_shape.hpp>
# include <pika/type_support/pack.hpp>

# include <exception>
# include <iterator>
# include <tuple>
# include <type_traits>
# include <utility>

namespace pika::bulk_detail {
    template <typename Sender, typename Shape, typename F>
    struct bulk_sender_impl
    {
        struct bulk_sender_type;
    };

    template <typename Sender, typename Shape, typename F>
    using bulk_sender = typename bulk_sender_impl<Sender, Shape, F>::bulk_sender_type;

    template <typename Sender, typename Shape, typename F>
    struct bulk_sender_impl<Sender, Shape, F>::bulk_sender_type
    {
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<Shape> shape;
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;

        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types = typename pika::execution::experimental::sender_traits<
            Sender>::template value_types<Tuple, Variant>;

        template <template <typename...> class Variant>
        using error_types = pika::util::detail::unique_t<
            pika::util::detail::prepend_t<typename pika::execution::experimental::sender_traits<
                                              Sender>::template error_types<Variant>,
                std::exception_ptr>>;

        static constexpr bool sends_done = false;

        friend constexpr decltype(auto) tag_invoke(
            pika::execution::experimental::get_env_t, bulk_sender_type const& s) noexcept
        {
            return pika::execution::experimental::get_env(s.sender);
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
            friend void tag_invoke(pika::execution::experimental::set_error_t, bulk_receiver&& r,
                Error&& error) noexcept
            {
                pika::execution::experimental::set_error(
                    std::move(r.receiver), PIKA_FORWARD(Error, error));
            }

            friend void tag_invoke(
                pika::execution::experimental::set_stopped_t, bulk_receiver&& r) noexcept
            {
                pika::execution::experimental::set_stopped(std::move(r.receiver));
            }

            template <typename... Ts>
            void set_value(Ts&&... ts) && noexcept
            {
                auto r = std::move(*this);
                pika::detail::try_catch_exception_ptr(
                    [&]() {
                        for (auto const& s : r.shape) { PIKA_INVOKE(r.f, s, ts...); }
                        pika::execution::experimental::set_value(
                            std::move(r.receiver), PIKA_FORWARD(Ts, ts)...);
                    },
                    [&](std::exception_ptr ep) {
                        pika::execution::experimental::set_error(
                            std::move(r.receiver), std::move(ep));
                    });
            }
        };

        template <typename Receiver>
        friend auto tag_invoke(
            pika::execution::experimental::connect_t, bulk_sender_type&& s, Receiver&& receiver)
        {
            return pika::execution::experimental::connect(std::move(s.sender),
                bulk_receiver<Receiver>(
                    PIKA_FORWARD(Receiver, receiver), std::move(s.shape), std::move(s.f)));
        }

        template <typename Receiver>
        friend auto tag_invoke(pika::execution::experimental::connect_t, bulk_sender_type const& s,
            Receiver&& receiver)
        {
            return pika::execution::experimental::connect(
                s.sender, bulk_receiver<Receiver>(PIKA_FORWARD(Receiver, receiver), s.shape, s.f));
        }
    };
}    // namespace pika::bulk_detail

namespace pika::execution::experimental {
    inline constexpr struct bulk_t final : pika::functional::detail::tag_priority<bulk_t>
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
        friend constexpr PIKA_FORCEINLINE auto
        tag_override_invoke(bulk_t, Sender&& sender, Shape const& shape, F&& f)
        {
            auto scheduler = pika::execution::experimental::get_completion_scheduler<
                pika::execution::experimental::set_value_t>(
                pika::execution::experimental::get_env(sender));
            return pika::functional::detail::tag_invoke(bulk_t{}, std::move(scheduler),
                PIKA_FORWARD(Sender, sender), shape, PIKA_FORWARD(F, f));
        }

        // clang-format off
        template <typename Sender, typename Shape, typename F,
            PIKA_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                std::is_integral<Shape>::value
            )>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(bulk_t, Sender&& sender, Shape const& shape, F&& f)
        {
            return bulk_detail::bulk_sender<Sender, pika::util::detail::counting_shape_type<Shape>,
                F>{PIKA_FORWARD(Sender, sender), pika::util::detail::make_counting_shape(shape),
                PIKA_FORWARD(F, f)};
        }

        // clang-format off
        template <typename Sender, typename Shape, typename F,
            PIKA_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                !std::is_integral<std::decay_t<Shape>>::value
            )>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(bulk_t, Sender&& sender, Shape&& shape, F&& f)
        {
            return bulk_detail::bulk_sender<Sender, Shape, F>{
                PIKA_FORWARD(Sender, sender), PIKA_FORWARD(Shape, shape), PIKA_FORWARD(F, f)};
        }

        template <typename Shape, typename F>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(bulk_t, Shape&& shape, F&& f)
        {
            return detail::partial_algorithm<bulk_t, Shape, F>{
                PIKA_FORWARD(Shape, shape), PIKA_FORWARD(F, f)};
        }
    } bulk{};
}    // namespace pika::execution::experimental
#endif
