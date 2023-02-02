//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#include <pika/allocator_support/allocator_deleter.hpp>
#include <pika/allocator_support/internal_allocator.hpp>
#include <pika/allocator_support/traits/is_allocator.hpp>
#include <pika/assert.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/datastructures/variant.hpp>
#include <pika/execution/algorithms/detail/helpers.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution_base/operation_state.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/unique_function.hpp>
#include <pika/memory/intrusive_ptr.hpp>
#include <pika/synchronization/spinlock.hpp>
#include <pika/thread_support/atomic_count.hpp>
#include <pika/type_support/detail/with_result_of.hpp>
#include <pika/type_support/pack.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pika::split_tuple_detail {
    template <typename Receiver>
    struct error_visitor
    {
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

        template <typename Error>
        void operator()(Error const& error)
        {
            pika::execution::experimental::set_error(
                PIKA_MOVE(receiver), error);
        }
    };

    template <typename Sender, typename Allocator>
    struct shared_state
    {
        struct split_tuple_receiver;

        using allocator_type = typename std::allocator_traits<
            Allocator>::template rebind_alloc<shared_state>;
        PIKA_NO_UNIQUE_ADDRESS allocator_type alloc;
        using mutex_type = pika::spinlock;
        mutex_type mtx;
        pika::detail::atomic_count reference_count{0};
        std::atomic<bool> start_called{false};
        std::atomic<bool> predecessor_done{false};

        using operation_state_type =
            std::decay_t<pika::execution::experimental::connect_result_t<Sender,
                split_tuple_receiver>>;
        // We store the operation state in an optional so that we can reset
        // it as soon as the the split_tuple_receiver has been signaled.
        // This is useful to ensure that resources held by the predecessor
        // work is released as soon as possible.
        std::optional<operation_state_type> os;

        // nvcc does not like decay_t, so this uses decay<>::type instead.
#if defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
        using value_type = typename std::decay<
            pika::execution::experimental::detail::single_result_t<
                typename pika::execution::experimental::value_types_of_t<Sender,
                    pika::execution::experimental::detail::empty_env,
                    pika::util::detail::pack, pika::util::detail::pack>>>::type;
#else
        using value_type = typename std::decay<
            pika::execution::experimental::detail::single_result_t<
                typename pika::execution::experimental::sender_traits<
                    Sender>::template value_types<pika::util::detail::pack,
                    pika::util::detail::pack>>>::type;
#endif

#if defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
        using error_type =
            pika::util::detail::unique_t<pika::util::detail::prepend_t<
                pika::util::detail::transform_t<
                    typename pika::execution::experimental::error_types_of_t<
                        Sender,
                        pika::execution::experimental::detail::empty_env,
                        pika::detail::variant>,
                    std::decay>,
                std::exception_ptr>>;
#else
        using error_type =
            pika::util::detail::unique_t<pika::util::detail::prepend_t<
                pika::util::detail::transform_t<
                    typename pika::execution::experimental::sender_traits<
                        Sender>::template error_types<pika::detail::variant>,
                    std::decay>,
                std::exception_ptr>>;
#endif
        pika::detail::variant<pika::detail::monostate,
            pika::execution::detail::stopped_type, error_type, value_type>
            v;

        using continuation_type = pika::util::detail::unique_function<void()>;

        std::array<continuation_type, std::tuple_size_v<value_type>>
            continuations;

        struct split_tuple_receiver
        {
            shared_state& state;

            template <typename Error>
            friend void tag_invoke(pika::execution::experimental::set_error_t,
                split_tuple_receiver&& r, Error&& error) noexcept
            {
                r.state.v.template emplace<error_type>(
                    error_type(PIKA_FORWARD(Error, error)));
                r.state.set_predecessor_done();
            }

            friend void tag_invoke(pika::execution::experimental::set_stopped_t,
                split_tuple_receiver&& r) noexcept
            {
                r.state.set_predecessor_done();
            };

                // This typedef is duplicated from the parent struct. The
                // parent typedef is not instantiated early enough for use
                // here.
#if defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
            using value_type = typename std::decay<
                pika::execution::experimental::detail::single_result_t<
                    typename pika::execution::experimental::value_types_of_t<
                        Sender,
                        pika::execution::experimental::detail::empty_env,
                        pika::util::detail::pack, pika::util::detail::pack>>>::
                type;
#else
            using value_type = typename std::decay<
                pika::execution::experimental::detail::single_result_t<
                    typename pika::execution::experimental::sender_traits<
                        Sender>::template value_types<pika::util::detail::pack,
                        pika::util::detail::pack>>>::type;
#endif

            template <typename T>
            friend auto tag_invoke(pika::execution::experimental::set_value_t,
                split_tuple_receiver&& r, T&& t) noexcept
                -> decltype(std::declval<pika::detail::variant<
                                pika::detail::monostate, value_type>>()
                                .template emplace<value_type>(
                                    PIKA_FORWARD(T, t)),
                    void())
            {
                r.state.v.template emplace<value_type>(PIKA_FORWARD(T, t));

                r.state.set_predecessor_done();
            }

            friend constexpr pika::execution::experimental::detail::empty_env
            tag_invoke(pika::execution::experimental::get_env_t,
                split_tuple_receiver const&) noexcept
            {
                return {};
            }
        };

        template <typename Sender_,
            typename = std::enable_if_t<
                !std::is_same<std::decay_t<Sender_>, shared_state>::value>>
        shared_state(Sender_&& sender, allocator_type const& alloc)
          : alloc(alloc)
        {
            os.emplace(pika::detail::with_result_of([&]() {
                return pika::execution::experimental::connect(
                    PIKA_FORWARD(Sender_, sender), split_tuple_receiver{*this});
            }));
        }

        ~shared_state()
        {
            PIKA_ASSERT_MSG(start_called,
                "start was never called on the operation state of split_tuple. "
                "Did you forget to connect the sender to a receiver, or call "
                "start on the operation state?");
        }

        template <std::size_t Index, typename Receiver>
        struct stopped_error_value_visitor
        {
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

            [[noreturn]] void operator()(pika::detail::monostate) const
            {
                PIKA_UNREACHABLE;
            }

            void operator()(pika::execution::detail::stopped_type)
            {
                constexpr bool sends_stopped =
#if defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
                    pika::execution::experimental::sends_stopped<Sender,
                        pika::execution::experimental::detail::empty_env>
#else
                    pika::execution::experimental::sender_traits<
                        Sender>::sends_done
#endif
                    ;
                if constexpr (sends_stopped)
                {
                    pika::execution::experimental::set_stopped(
                        PIKA_MOVE(receiver));
                }
                else
                {
                    PIKA_UNREACHABLE;
                }
            }

            void operator()(error_type const& error)
            {
                pika::detail::visit(
                    error_visitor<Receiver>{PIKA_FORWARD(Receiver, receiver)},
                    error);
            }

            void operator()(value_type& t)
            {
                pika::execution::experimental::set_value(
                    PIKA_MOVE(receiver), std::move(std::get<Index>(t)));
            }
        };

        void set_predecessor_done()
        {
            // We reset the operation state as soon as the predecessor
            // is done to release any resources held by it. Any values
            // sent by the predecessor have already been stored in the
            // shared state by now.
            os.reset();

            predecessor_done = true;

            {
                // We require taking the lock here to synchronize with
                // threads attempting to add continuations to the vector
                // of continuations. However, it is enough to take it
                // once and release it immediately.
                //
                // Without the lock we may not see writes to the vector.
                // With the lock threads attempting to add continuations
                // will either:
                // - See predecessor_done = true in which case they will
                //   call the continuation directly without adding it to
                //   the vector of continuations. Accessing the vector
                //   below without the lock is safe in this case because
                //   the vector is not modified.
                // - See predecessor_done = false and proceed to take
                //   the lock. If they see predecessor_done after taking
                //   the lock they can again release the lock and call
                //   the continuation directly. Accessing the vector
                //   without the lock is again safe because the vector
                //   is not modified.
                // - See predecessor_done = false and proceed to take
                //   the lock. If they see predecessor_done is still
                //   false after taking the lock, they will proceed to
                //   add a continuation to the vector. Since they keep
                //   the lock they can safely write to the vector. This
                //   thread will not proceed past the lock until they
                //   have finished writing to the vector.
                //
                // Importantly, once this thread has taken and released
                // this lock, threads attempting to add continuations to
                // the vector must see predecessor_done = true after
                // taking the lock in their threads and will not add
                // continuations to the vector.
                std::unique_lock<mutex_type> l{mtx};
            }

            if (!continuations.empty())
            {
                // We move the continuations to a local variable to
                // release them once all continuations have been called.
                // We cannot call clear on continuations after the loop
                // because the shared state may already be released by
                // the last continuation to run.
                auto continuations_local = PIKA_MOVE(continuations);
                for (auto const& continuation : continuations_local)
                {
                    if (continuation)
                    {
                        continuation();
                    }
                }
            }
        }

        template <std::size_t Index, typename Receiver>
        void add_continuation(Receiver& receiver) = delete;

        template <std::size_t Index, typename Receiver>
        void add_continuation(Receiver&& receiver)
        {
            if (predecessor_done)
            {
                // If we read predecessor_done here it means that one of
                // set_error/set_stopped/set_value has been called and
                // values/errors have been stored into the shared state.
                // We can trigger the continuation directly.
                pika::detail::visit(
                    stopped_error_value_visitor<Index, Receiver>{
                        PIKA_FORWARD(Receiver, receiver)},
                    v);
            }
            else
            {
                // If predecessor_done is false, we have to take the
                // lock to potentially add the continuation to the
                // vector of continuations.
                std::unique_lock<mutex_type> l{mtx};

                if (predecessor_done)
                {
                    // By the time the lock has been taken,
                    // predecessor_done might already be true and we can
                    // release the lock early and call the continuation
                    // directly again.
                    l.unlock();
                    pika::detail::visit(
                        stopped_error_value_visitor<Index, Receiver>{
                            PIKA_FORWARD(Receiver, receiver)},
                        v);
                }
                else
                {
                    // If predecessor_done is still false, we add the
                    // continuation to the vector of continuations. This
                    // has to be done while holding the lock, since
                    // other threads may also try to add continuations
                    // to the vector and the vector is not threadsafe in
                    // itself. The continuation will be called later
                    // when set_error/set_stopped/set_value is called.
                    continuations[Index] = [this,
                                               receiver = PIKA_FORWARD(Receiver,
                                                   receiver)]() mutable {
                        pika::detail::visit(
                            stopped_error_value_visitor<Index, Receiver>{
                                PIKA_MOVE(receiver)},
                            v);
                    };
                }
            }
        }

        void start() & noexcept
        {
            if (!start_called.exchange(true))
            {
                PIKA_ASSERT(os.has_value());
                pika::execution::experimental::start(os.value());
            }
        }

        friend void intrusive_ptr_add_ref(shared_state* p)
        {
            ++p->reference_count;
        }

        friend void intrusive_ptr_release(shared_state* p)
        {
            if (--p->reference_count == 0)
            {
                allocator_type other_alloc(p->alloc);
                std::allocator_traits<allocator_type>::destroy(other_alloc, p);
                std::allocator_traits<allocator_type>::deallocate(
                    other_alloc, p, 1);
            }
        }
    };

    template <typename Sender, typename Allocator, std::size_t Index>
    struct split_tuple_sender_impl
    {
        struct split_tuple_sender_type;
    };

    template <typename Sender, typename Allocator, std::size_t Index>
    using split_tuple_sender = typename split_tuple_sender_impl<Sender,
        Allocator, Index>::split_tuple_sender_type;

    template <typename Sender, typename Allocator, std::size_t Index>
    struct split_tuple_sender_impl<Sender, Allocator,
        Index>::split_tuple_sender_type
    {
        using allocator_type = Allocator;
        using shared_state_type = shared_state<Sender, Allocator>;

        // nvcc does not like decay_t, so this uses decay<>::type instead.
#if defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
        using value_type = typename std::decay<
            pika::execution::experimental::detail::single_result_t<
                typename pika::execution::experimental::value_types_of_t<Sender,
                    pika::execution::experimental::detail::empty_env,
                    pika::util::detail::pack, pika::util::detail::pack>>>::type;
#else
        using value_type = typename std::decay<
            pika::execution::experimental::detail::single_result_t<
                typename pika::execution::experimental::sender_traits<
                    Sender>::template value_types<pika::util::detail::pack,
                    pika::util::detail::pack>>>::type;
#endif
        static_assert(std::tuple_size_v<value_type> >= 1,
            "split_tuple takes a sender that sends a tuple of at least one "
            "type");
        using split_tuple_sender_value_type =
            std::decay_t<std::tuple_element_t<Index, value_type>>;

        template <typename T>
        struct add_const_lvalue_reference
        {
            using type = std::add_lvalue_reference_t<std::add_const_t<T>>;
        };

#if defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
        template <typename...>
        using set_value_helper =
            pika::execution::experimental::completion_signatures<
                pika::execution::experimental::set_value_t(
                    split_tuple_sender_value_type)>;

        template <typename T>
        using set_error_helper =
            pika::execution::experimental::completion_signatures<
                pika::execution::experimental::set_error_t(
                    typename add_const_lvalue_reference<T>::type)>;

        using completion_signatures =
            pika::execution::experimental::make_completion_signatures<Sender,
                pika::execution::experimental::detail::empty_env,
                pika::execution::experimental::completion_signatures<
                    pika::execution::experimental::set_error_t(
                        std::exception_ptr)>,
                set_value_helper, set_error_helper>;
#else
        template <template <typename...> class Tuple,
            template <typename...> class Variant>
        using value_types = Variant<Tuple<split_tuple_sender_value_type>>;

        template <template <typename...> class Variant>
        using error_types =
            pika::util::detail::unique_t<pika::util::detail::prepend_t<
                pika::util::detail::transform_t<
                    typename pika::execution::experimental::sender_traits<
                        Sender>::template error_types<Variant>,
                    add_const_lvalue_reference>,
                std::exception_ptr>>;

        static constexpr bool sends_done =
            pika::execution::experimental::sender_traits<Sender>::sends_done;
#endif

        pika::intrusive_ptr<shared_state_type> state;

        explicit split_tuple_sender_type(
            pika::intrusive_ptr<shared_state_type> state)
          : state(std::move(state))
        {
        }

        split_tuple_sender_type(split_tuple_sender_type const&) = default;
        split_tuple_sender_type& operator=(
            split_tuple_sender_type const&) = default;
        split_tuple_sender_type(split_tuple_sender_type&&) = default;
        split_tuple_sender_type& operator=(split_tuple_sender_type&&) = default;

        template <typename Receiver>
        struct operation_state
        {
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            pika::intrusive_ptr<shared_state_type> state;

            template <typename Receiver_>
            operation_state(Receiver_&& receiver,
                pika::intrusive_ptr<shared_state_type> state)
              : receiver(PIKA_FORWARD(Receiver_, receiver))
              , state(PIKA_MOVE(state))
            {
            }

            operation_state(operation_state&&) = delete;
            operation_state& operator=(operation_state&&) = delete;
            operation_state(operation_state const&) = delete;
            operation_state& operator=(operation_state const&) = delete;

            friend void tag_invoke(pika::execution::experimental::start_t,
                operation_state& os) noexcept
            {
                os.state->start();
                os.state->template add_continuation<Index>(
                    PIKA_MOVE(os.receiver));
            }
        };

        template <typename Receiver>
        friend operation_state<Receiver>
        tag_invoke(pika::execution::experimental::connect_t,
            split_tuple_sender_type&& s, Receiver&& receiver)
        {
            return {PIKA_FORWARD(Receiver, receiver), PIKA_MOVE(s.state)};
        }
    };

    template <typename Sender, typename Allocator, std::size_t... Is>
    auto make_split_tuple_senders(
        pika::intrusive_ptr<shared_state<Sender, Allocator>> state,
        pika::util::detail::index_pack<Is...>)
    {
        return std::tuple(split_tuple_sender<Sender, Allocator, Is>(state)...);
    }

    template <typename Sender, typename Allocator>
    auto make_split_tuple_senders(Sender&& sender, Allocator const& allocator)
    {
        using allocator_type = std::decay_t<Allocator>;
        using sender_type = std::decay_t<Sender>;
        using shared_state_type = shared_state<sender_type, allocator_type>;
        using other_allocator = typename std::allocator_traits<
            allocator_type>::template rebind_alloc<shared_state_type>;
        using allocator_traits = std::allocator_traits<other_allocator>;
        using unique_ptr = std::unique_ptr<shared_state_type,
            pika::detail::allocator_deleter<other_allocator>>;

        other_allocator alloc(allocator);
        unique_ptr p(allocator_traits::allocate(alloc, 1),
            pika::detail::allocator_deleter<other_allocator>{alloc});

        new (p.get())
            shared_state_type{PIKA_FORWARD(Sender_, sender), allocator};
        pika::intrusive_ptr<shared_state_type> state = p.release();

        // nvcc does not like decay_t, so this uses decay<>::type instead.
#if defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
        using value_type = typename std::decay<
            pika::execution::experimental::detail::single_result_t<
                typename pika::execution::experimental::value_types_of_t<Sender,
                    pika::execution::experimental::detail::empty_env,
                    pika::util::detail::pack, pika::util::detail::pack>>>::type;
#else
        using value_type = typename std::decay<
            pika::execution::experimental::detail::single_result_t<
                typename pika::execution::experimental::sender_traits<
                    Sender>::template value_types<pika::util::detail::pack,
                    pika::util::detail::pack>>>::type;
#endif

        // nvcc does not like tuple_size_v, so this uses tuple_size<>::value
        // instead.
        return make_split_tuple_senders<std::decay_t<Sender>,
            std::decay_t<Allocator>>(std::move(state),
            pika::util::detail::make_index_pack_t<
                std::tuple_size<value_type>::value>{});
    }
}    // namespace pika::split_tuple_detail

namespace pika::execution::experimental {
    /// \brief Splits a sender of a tuple into a tuple of senders.
    ///
    /// Sender adaptor that takes a sender that sends a single, non-empty, tuple
    /// and returns a new tuple of the same size as the one sent by the input
    /// sender which contains one sender for each element in the input sender
    /// tuple. Each output sender signals completion whenever the input sender
    /// would have signalled completion.
    inline constexpr struct split_tuple_t final
      : pika::functional::detail::tag_fallback<split_tuple_t>
    {
    private:
        template <typename Sender, PIKA_CONCEPT_REQUIRES_(is_sender_v<Sender>)>
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(split_tuple_t, Sender&& sender)
        {
            return split_tuple_detail::make_split_tuple_senders(
                PIKA_FORWARD(Sender, sender),
                pika::detail::internal_allocator<>{});
        }

        template <typename Sender, typename Allocator,
            PIKA_CONCEPT_REQUIRES_(
                is_sender_v<Sender>&& pika::detail::is_allocator_v<Allocator>)>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            split_tuple_t, Sender&& sender, Allocator const& allocator)
        {
            return split_tuple_detail::make_split_tuple_senders(
                PIKA_FORWARD(Sender, sender), allocator);
        }

        template <typename Allocator = pika::detail::internal_allocator<>,
            PIKA_CONCEPT_REQUIRES_(pika::detail::is_allocator_v<Allocator>)>
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(split_tuple_t, Allocator const& allocator = {})
        {
            return detail::partial_algorithm<split_tuple_t, Allocator>{
                allocator};
        }
    } split_tuple{};
}    // namespace pika::execution::experimental
