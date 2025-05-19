//  Copyright (c) 2020 ETH Zurich
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
# include <pika/errors/try_catch_exception_ptr.hpp>
# include <pika/execution/algorithms/detail/partial_algorithm.hpp>
# include <pika/execution_base/completion_scheduler.hpp>
# include <pika/execution_base/receiver.hpp>
# include <pika/execution_base/sender.hpp>
# include <pika/functional/detail/invoke.hpp>
# include <pika/functional/detail/tag_fallback_invoke.hpp>
# include <pika/type_support/pack.hpp>

# include <exception>
# include <type_traits>
# include <utility>

namespace pika::then_detail {
    template <typename Receiver, typename F>
    struct then_receiver_impl
    {
        struct then_receiver_type;
    };

    template <typename Receiver, typename F>
    using then_receiver = typename then_receiver_impl<Receiver, F>::then_receiver_type;

    template <typename Receiver, typename F>
    struct then_receiver_impl<Receiver, F>::then_receiver_type
    {
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;

        template <typename Error>
        void set_error(Error&& error) && noexcept
        {
            auto r = std::move(*this);
            pika::execution::experimental::set_error(
                std::move(r.receiver), std::forward<Error>(error));
        }

        friend void tag_invoke(
            pika::execution::experimental::set_stopped_t, then_receiver_type&& r) noexcept
        {
            pika::execution::experimental::set_stopped(std::move(r.receiver));
        }

        template <typename... Ts>
        void set_value(Ts&&... ts) && noexcept
        {
            auto r = std::move(*this);
            pika::detail::try_catch_exception_ptr(
                [&]() {
                    if constexpr (std::is_void_v<std::invoke_result_t<F, Ts...>>)
                    {
                    // Certain versions of GCC with optimizations fail on
                    // the move with an internal compiler error.
# if defined(PIKA_GCC_VERSION) && (PIKA_GCC_VERSION < 100000)
                        PIKA_INVOKE(std::move(r.f), std::forward<Ts>(ts)...);
# else
                        PIKA_INVOKE(std::move(r.f), std::forward<Ts>(ts)...);
# endif
                        pika::execution::experimental::set_value(std::move(r.receiver));
                    }
                    else
                    {
                    // Certain versions of GCC with optimizations fail on
                    // the move with an internal compiler error.
# if defined(PIKA_GCC_VERSION) && (PIKA_GCC_VERSION < 100000)
                        pika::execution::experimental::set_value(std::move(r.receiver),
                            PIKA_INVOKE(std::move(r.f), std::forward<Ts>(ts)...));
# else
                        pika::execution::experimental::set_value(std::move(r.receiver),
                            PIKA_INVOKE(std::move(r.f), std::forward<Ts>(ts)...));
# endif
                    }
                },
                [&](std::exception_ptr ep) {
                    pika::execution::experimental::set_error(std::move(r.receiver), std::move(ep));
                });
        }
    };

    template <typename Sender, typename F>
    struct then_sender
    {
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;

        template <typename Tuple>
        struct invoke_result_helper;

        template <template <typename...> class Tuple, typename... Ts>
        struct invoke_result_helper<Tuple<Ts...>>
        {
            using result_type = std::invoke_result_t<F, Ts...>;
            using type =
                std::conditional_t<std::is_void<result_type>::value, Tuple<>, Tuple<result_type>>;
        };

        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types = pika::util::detail::unique_t<
            pika::util::detail::transform_t<typename pika::execution::experimental::sender_traits<
                                                Sender>::template value_types<Tuple, Variant>,
                invoke_result_helper>>;

        template <template <typename...> class Variant>
        using error_types = pika::util::detail::unique_t<
            pika::util::detail::prepend_t<typename pika::execution::experimental::sender_traits<
                                              Sender>::template error_types<Variant>,
                std::exception_ptr>>;

        static constexpr bool sends_done = false;

        template <typename Receiver>
        auto connect(Receiver&& receiver) &&
        {
            return pika::execution::experimental::connect(std::move(sender),
                then_receiver<Receiver, F>{std::forward<Receiver>(receiver), std::move(f)});
        }

        template <typename Receiver>
        auto connect(Receiver&& receiver) const&
        {
            return pika::execution::experimental::connect(
                sender, then_receiver<Receiver, F>{std::forward<Receiver>(receiver), f});
        }

        decltype(auto) get_env() const& noexcept
        {
            return pika::execution::experimental::get_env(sender);
        }
    };
}    // namespace pika::then_detail

namespace pika::execution::experimental {
    inline constexpr struct then_t final : pika::functional::detail::tag_fallback<then_t>
    {
    private:
        // clang-format off
        template <typename Sender, typename F,
            PIKA_CONCEPT_REQUIRES_(
                is_sender_v<Sender>
            )>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(then_t, Sender&& sender, F&& f)
        {
            return then_detail::then_sender<Sender, F>{
                std::forward<Sender>(sender), std::forward<F>(f)};
        }

        template <typename F>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(then_t, F&& f)
        {
            return detail::partial_algorithm<then_t, F>{std::forward<F>(f)};
        }
    } then{};
}    // namespace pika::execution::experimental
#endif
