//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_STDEXEC)
# include <pika/execution_base/stdexec_forward.hpp>
#endif

#include <pika/assert.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/datastructures/variant.hpp>
#include <pika/execution/algorithms/detail/helpers.hpp>
#include <pika/execution_base/operation_state.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/type_support/detail/with_result_of.hpp>
#include <pika/type_support/pack.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika::when_all_vector_detail {
    // This is a helper empty type for the case that nothing is sent
    // from the predecessors
    struct void_value_type
    {
    };

    template <typename Sender>
    struct type_helper
    {
        // We expect a single value type or nothing from the predecessor
        // sender type
#if defined(PIKA_HAVE_STDEXEC)
        using element_value_type =
            std::decay_t<pika::execution::experimental::detail::single_result_t<pika::execution::
                    experimental::value_types_of_t<Sender, pika::execution::experimental::empty_env,
                        pika::util::detail::pack, pika::util::detail::pack>>>;

        // This sender sends any error types sent by the predecessor senders
        // or std::exception_ptr
        template <template <typename...> class Variant>
        using error_types = pika::util::detail::unique_concat_t<
            pika::util::detail::transform_t<pika::execution::experimental::error_types_of_t<Sender,
                                                pika::execution::experimental::empty_env, Variant>,
                std::decay>,
            Variant<std::exception_ptr>>;
#else
        using element_value_type =
            std::decay_t<pika::execution::experimental::detail::single_result_t<
                typename pika::execution::experimental::sender_traits<Sender>::template value_types<
                    pika::util::detail::pack, pika::util::detail::pack>>>;

        // This sender sends any error types sent by the predecessor senders
        // or std::exception_ptr
        template <template <typename...> class Variant>
        using error_types = pika::util::detail::unique_concat_t<
            pika::util::detail::transform_t<typename pika::execution::experimental::sender_traits<
                                                Sender>::template error_types<Variant>,
                std::decay>,
            Variant<std::exception_ptr>>;
#endif
        static constexpr bool is_void_value_type = std::is_void_v<element_value_type>;
    };

    template <typename Sender, typename Receiver>
    struct operation_state
    {
        struct when_all_vector_receiver
        {
            PIKA_STDEXEC_RECEIVER_CONCEPT

            operation_state& op_state;
            std::size_t const i;

            template <typename Error>
            void set_error(Error&& error) noexcept
            {
                auto r = std::move(*this);
                if (!r.op_state.set_stopped_error_called.exchange(true))
                {
                    try
                    {
                        r.op_state.error = std::forward<Error>(error);
                    }
                    catch (...)
                    {
                        // NOLINTNEXTLINE(bugprone-throw-keyword-missing)
                        r.op_state.error = std::current_exception();
                    }
                }

                r.op_state.finish();
            }

            friend void tag_invoke(
                pika::execution::experimental::set_stopped_t, when_all_vector_receiver&& r) noexcept
            {
                r.op_state.set_stopped_error_called = true;
                r.op_state.finish();
            };

            template <typename... Ts>
            void set_value(Ts&&... ts) && noexcept
            {
                auto r = std::move(*this);
                if (!r.op_state.set_stopped_error_called)
                {
                    try
                    {
                        // We only have something to store if the
                        // predecessor sends the single value that it
                        // should send. We have nothing to store for
                        // predecessor senders that send nothing.
                        if constexpr (sizeof...(Ts) == 1)
                        {
                            r.op_state.ts[r.i].emplace(std::forward<Ts>(ts)...);
                        }
                    }
                    catch (...)
                    {
                        if (!r.op_state.set_stopped_error_called.exchange(true))
                        {
                            // NOLINTNEXTLINE(bugprone-throw-keyword-missing)
                            r.op_state.error = std::current_exception();
                        }
                    }
                }

                r.op_state.finish();
            }

            constexpr pika::execution::experimental::empty_env get_env() const& noexcept
            {
                return {};
            }
        };

        std::size_t num_predecessors;
        Receiver receiver;

        // Number of predecessor senders that have not yet called any of
        // the set signals.
        std::atomic<std::size_t> predecessors_remaining{num_predecessors};

        // The values sent by the predecessor senders are stored in a
        // vector of optional or the dummy type void_value_type if the
        // predecessor senders send nothing
        using types = type_helper<Sender>;
        using value_types_storage_type = std::conditional_t<types::is_void_value_type,
            void_value_type, std::vector<std::optional<typename types::element_value_type>>>;
        value_types_storage_type ts;

        // The first error sent by any predecessor sender is stored in a
        // optional of a variant of the error_types
        using error_types_storage_type =
            std::optional<typename types::template error_types<pika::detail::variant>>;
        error_types_storage_type error;

        // Set to true when set_stopped or set_error has been called
        std::atomic<bool> set_stopped_error_called{false};

        // The operation states are stored in an array of optionals of
        // the operation states to handle the non-movability and
        // non-copyability of them
        using operation_state_type =
            pika::execution::experimental::connect_result_t<Sender, when_all_vector_receiver>;
        using operation_states_storage_type =
            std::unique_ptr<std::optional<operation_state_type>[]>;
        operation_states_storage_type op_states = nullptr;

        template <typename Receiver_>
        operation_state(Receiver_&& receiver, std::vector<Sender> senders)
          : num_predecessors(senders.size())
          , receiver(std::forward<Receiver_>(receiver))
          , predecessors_remaining{num_predecessors}
        {
            op_states = std::make_unique<std::optional<operation_state_type>[]>(num_predecessors);
            std::size_t i = 0;
            for (auto& sender : senders)
            {
                op_states[i].emplace(pika::detail::with_result_of([&]() {
                    return pika::execution::experimental::connect(
#if defined(__NVCC__) && defined(PIKA_CUDA_VERSION) && (PIKA_CUDA_VERSION >= 1204)
                        std::move(sender)
#else
                        std::move(sender)
#endif
                            ,
                        when_all_vector_receiver{*this, i});
                }));
                ++i;
            }

            if constexpr (!types::is_void_value_type) { ts.resize(num_predecessors); }
        }

        operation_state(operation_state&&) = delete;
        operation_state& operator=(operation_state&&) = delete;
        operation_state(operation_state const&) = delete;
        operation_state& operator=(operation_state const&) = delete;

        void finish() noexcept
        {
            if (--predecessors_remaining == 0)
            {
                if (!set_stopped_error_called)
                {
                    if constexpr (types::is_void_value_type)
                    {
                        pika::execution::experimental::set_value(std::move(receiver));
                    }
                    else
                    {
                        std::vector<typename types::element_value_type> values;
                        values.reserve(num_predecessors);
                        for (auto&& t : ts)
                        {
                            PIKA_ASSERT(t.has_value());
                            values.push_back(
#if defined(__NVCC__) && defined(PIKA_CUDA_VERSION) && (PIKA_CUDA_VERSION >= 1204)
                                // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                                std::move(*t)
#else
                                // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                                std::move(*t)
#endif
                            );
                        }
                        pika::execution::experimental::set_value(
                            std::move(receiver), std::move(values));
                    }
                }
                else if (error)
                {
                    pika::detail::visit(
                        [this](auto&& error) {
                            pika::execution::experimental::set_error(
                                std::move(receiver), std::forward<decltype(error)>(error));
                        },
                        std::move(*error));
                }
                else
                {
#if defined(PIKA_HAVE_STDEXEC)
                    if constexpr (pika::execution::experimental::sends_stopped<Sender>)
#else
                    if constexpr (pika::execution::experimental::sender_traits<Sender>::sends_done)
#endif
                    {
                        pika::execution::experimental::set_stopped(std::move(receiver));
                    }
                    else { PIKA_UNREACHABLE; }
                }
            }
        }

        void start() & noexcept
        {
            // If there are no predecessors we can signal the
            // continuation as soon as start is called.
            if (num_predecessors == 0)
            {
                // If the predecessor sender type sends nothing, we also
                // send nothing to the continuation.
                if constexpr (types::is_void_value_type)
                {
                    pika::execution::experimental::set_value(std::move(receiver));
                }
                // If the predecessor sender type sends something we
                // send an empty vector of that type to the continuation.
                else
                {
                    pika::execution::experimental::set_value(
                        std::move(receiver), std::vector<typename types::element_value_type>{});
                }
            }
            // Otherwise we start all the operation states and wait for
            // the predecessors to signal completion.
            else
            {
                // After the call to start on the last child operation state the current
                // when_all_vector operation state may already have been released. We read the
                // number of predecessors from the operation state into a stack-local variable
                // so that the loop can end without reading freed memory.
                std::size_t const num_predecessors_local = num_predecessors;
                for (std::size_t i = 0; i < num_predecessors_local; ++i)
                {
                    PIKA_ASSERT(op_states[i].has_value());
                    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                    pika::execution::experimental::start(*(op_states.get()[i]));
                }
            }
        }
    };

    template <typename Sender>
    struct when_all_vector_sender
    {
        PIKA_STDEXEC_SENDER_CONCEPT

        using senders_type = std::vector<Sender>;
        senders_type senders;

        explicit constexpr when_all_vector_sender(senders_type&& senders)
          : senders(std::move(senders))
        {
        }

        explicit constexpr when_all_vector_sender(senders_type const& senders)
          : senders(senders)
        {
        }

        using types = type_helper<Sender>;

#if defined(PIKA_HAVE_STDEXEC)
        // This sender sends a single vector of the type sent by the
        // predecessor senders or nothing if the predecessor senders send
        // nothing
        template <typename...>
        using set_value_helper =
            pika::execution::experimental::completion_signatures<std::conditional_t<
                types::is_void_value_type, pika::execution::experimental::set_value_t(),
                pika::execution::experimental::set_value_t(
                    std::vector<typename types::element_value_type>)>>;

        static constexpr bool sends_done = false;

        using completion_signatures =
            pika::execution::experimental::transform_completion_signatures_of<Sender,
                pika::execution::experimental::empty_env,
                pika::execution::experimental::completion_signatures<
                    pika::execution::experimental::set_error_t(std::exception_ptr)>,
                set_value_helper>;
#else
        // This sender sends a single vector of the type sent by the
        // predecessor senders or nothing if the predecessor senders send
        // nothing
        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types = Variant<std::conditional_t<types::is_void_value_type, Tuple<>,
            Tuple<std::vector<typename types::element_value_type>>>>;

        template <template <typename...> class Variant>
        using error_types = typename types::template error_types<Variant>;

        static constexpr bool sends_done = false;
#endif

        template <typename Receiver>
        auto connect(Receiver&& receiver) &&
        {
            return operation_state<Sender, std::decay_t<Receiver>>(
                std::forward<Receiver>(receiver), std::move(senders));
        }

        template <typename Receiver>
        auto connect(Receiver&& receiver) const&
        {
            return operation_state<Sender, std::decay_t<Receiver>>(
                std::forward<Receiver>(receiver), senders);
        }
    };
}    // namespace pika::when_all_vector_detail

namespace pika::execution::experimental {
    struct when_all_vector_t final : pika::functional::detail::tag_fallback<when_all_vector_t>
    {
    private:
        template <typename Sender, PIKA_CONCEPT_REQUIRES_(is_sender_v<Sender>)>
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(when_all_vector_t, std::vector<Sender>&& senders)
        {
            return when_all_vector_detail::when_all_vector_sender<std::decay_t<Sender>>{
                std::move(senders)};
        }

        template <typename Sender, PIKA_CONCEPT_REQUIRES_(is_sender_v<Sender>)>
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(when_all_vector_t, std::vector<Sender> const& senders)
        {
            return when_all_vector_detail::when_all_vector_sender<std::decay_t<Sender>>{senders};
        }
    };

    /// \brief Returns a sender that completes when all senders in the input vector have completed.
    ///
    /// Sender adaptor that takes a vector of senders and returns a sender that sends a vector of
    /// the values sent by the input senders. The vector sent has the same size as the input vector.
    /// An empty vector of senders completes immediately on start. When the input vector of senders
    /// contains senders that send no value the output sender sends no value instead of a vector.
    /// The senders in the input vector must send at most a single type.
    inline constexpr when_all_vector_t when_all_vector{};
}    // namespace pika::execution::experimental
