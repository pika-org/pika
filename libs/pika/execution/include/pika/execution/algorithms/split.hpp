//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_STDEXEC)
# include <pika/execution_base/stdexec_forward.hpp>
#else
# include <pika/allocator_support/allocator_deleter.hpp>
# include <pika/allocator_support/internal_allocator.hpp>
# include <pika/allocator_support/traits/is_allocator.hpp>
# include <pika/assert.hpp>
# include <pika/concepts/concepts.hpp>
# include <pika/concurrency/spinlock.hpp>
# include <pika/datastructures/detail/small_vector.hpp>
# include <pika/datastructures/variant.hpp>
# include <pika/execution/algorithms/detail/helpers.hpp>
# include <pika/execution/algorithms/detail/partial_algorithm.hpp>
# include <pika/execution_base/operation_state.hpp>
# include <pika/execution_base/receiver.hpp>
# include <pika/execution_base/sender.hpp>
# include <pika/functional/bind_front.hpp>
# include <pika/functional/detail/tag_fallback_invoke.hpp>
# include <pika/functional/unique_function.hpp>
# include <pika/memory/intrusive_ptr.hpp>
# include <pika/thread_support/atomic_count.hpp>
# include <pika/type_support/detail/with_result_of.hpp>
# include <pika/type_support/pack.hpp>

# include <atomic>
# include <cstddef>
# include <exception>
# include <memory>
# include <mutex>
# include <optional>
# include <tuple>
# include <type_traits>
# include <utility>

namespace pika::split_detail {
    template <typename Receiver>
    struct error_visitor
    {
        std::decay_t<Receiver>& receiver;

        template <typename Error>
        void operator()(Error const& error)
        {
            pika::execution::experimental::set_error(std::move(receiver), error);
        }
    };

    template <typename Receiver>
    struct value_visitor
    {
        std::decay_t<Receiver>& receiver;

        template <typename Ts>
        void operator()(Ts const& ts)
        {
            std::apply(pika::util::detail::bind_front(
                           pika::execution::experimental::set_value, std::move(receiver)),
                ts);
        }
    };

    template <typename Sender, typename Allocator>
    struct split_sender_impl
    {
        struct split_sender_type;
    };

    template <typename Sender, typename Allocator>
    using split_sender = typename split_sender_impl<Sender, Allocator>::split_sender_type;

    template <typename Sender, typename Allocator>
    struct split_sender_impl<Sender, Allocator>::split_sender_type
    {
        struct split_sender_tag
        {
        };

        using allocator_type = Allocator;

        template <typename T>
        struct add_const_lvalue_reference
        {
            using type = std::add_lvalue_reference_t<std::add_const_t<T>>;
        };

        template <typename Tuple>
        struct value_types_helper
        {
            using type = pika::util::detail::transform_t<Tuple, add_const_lvalue_reference>;
        };

        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types =
            pika::util::detail::transform_t<typename pika::execution::experimental::sender_traits<
                                                Sender>::template value_types<Tuple, Variant>,
                value_types_helper>;

        template <template <typename...> class Variant>
        using error_types = pika::util::detail::unique_t<pika::util::detail::prepend_t<
            pika::util::detail::transform_t<typename pika::execution::experimental::sender_traits<
                                                Sender>::template error_types<Variant>,
                add_const_lvalue_reference>,
            std::exception_ptr>>;

        static constexpr bool sends_done = false;

        struct shared_state
        {
            struct split_receiver;

            using allocator_type =
                typename std::allocator_traits<Allocator>::template rebind_alloc<shared_state>;
            PIKA_NO_UNIQUE_ADDRESS allocator_type alloc;
            using mutex_type = pika::concurrency::detail::spinlock;
            mutex_type mtx;
            pika::detail::atomic_count reference_count{0};
            std::atomic<bool> start_called{false};
            std::atomic<bool> predecessor_done{false};

            using operation_state_type = std::decay_t<
                pika::execution::experimental::connect_result_t<Sender, split_receiver>>;
            // We store the operation state in an optional so that we can
            // reset it as soon as the the split_receiver has been signaled.
            // This is useful to ensure that resources held by the
            // predecessor work is released as soon as possible.
            std::optional<operation_state_type> os;

            template <typename Tuple>
            struct value_type_helper
            {
                using type = pika::util::detail::transform_t<Tuple, std::decay>;
            };

            using value_type = pika::util::detail::transform_t<
                typename pika::execution::experimental::sender_traits<Sender>::template value_types<
                    std::tuple, pika::detail::variant>,
                value_type_helper>;
            using error_type = pika::util::detail::unique_t<pika::util::detail::prepend_t<
                pika::util::detail::transform_t<
                    typename pika::execution::experimental::sender_traits<
                        Sender>::template error_types<pika::detail::variant>,
                    std::decay>,
                std::exception_ptr>>;
            pika::detail::variant<pika::detail::monostate, pika::execution::detail::stopped_type,
                error_type, value_type>
                v;

            using continuation_type = pika::util::detail::unique_function<void()>;
            pika::detail::small_vector<continuation_type, 1> continuations;

            struct split_receiver
            {
                pika::intrusive_ptr<shared_state> state;

                template <typename Error>
                friend void tag_invoke(pika::execution::experimental::set_error_t, split_receiver r,
                    Error&& error) noexcept
                {
                    r.state->v.template emplace<error_type>(error_type(std::forward<Error>(error)));
                    r.state->set_predecessor_done();
                }

                friend void tag_invoke(
                    pika::execution::experimental::set_stopped_t, split_receiver r) noexcept
                {
                    r.state->set_predecessor_done();
                };

                // This typedef is duplicated from the parent struct. The
                // parent typedef is not instantiated early enough for use
                // here.
                template <typename Tuple>
                struct value_type_helper
                {
                    using type = pika::util::detail::transform_t<Tuple, std::decay>;
                };
                using value_type = pika::util::detail::transform_t<
                    typename pika::execution::experimental::sender_traits<
                        Sender>::template value_types<std::tuple, pika::detail::variant>,
                    value_type_helper>;

                template <typename... Ts>
                auto set_value(Ts&&... ts) && noexcept
                    -> decltype(std::declval<
                                    pika::detail::variant<pika::detail::monostate, value_type>>()
                                    .template emplace<value_type>(
                                        std::make_tuple<>(std::forward<Ts>(ts)...)),
                        void())
                {
                    auto r = std::move(*this);
                    r.state->v.template emplace<value_type>(
                        std::make_tuple<>(std::forward<Ts>(ts)...));

                    r.state->set_predecessor_done();
                }
            };

            template <typename Sender_,
                typename =
                    std::enable_if_t<!std::is_same<std::decay_t<Sender_>, shared_state>::value>>
            shared_state(Sender_&& sender, allocator_type const& alloc)
              : alloc(alloc)
            {
                os.emplace(pika::detail::with_result_of([&]() {
                    return pika::execution::experimental::connect(
                        std::forward<Sender_>(sender), split_receiver{this});
                }));
            }

            ~shared_state()
            {
                PIKA_ASSERT_MSG(start_called,
                    "start was never called on the operation state of split. Did you forget to "
                    "connect the sender to a receiver, or call start on the operation state?");
            }

            template <typename Receiver>
            struct stopped_error_value_visitor
            {
                std::decay_t<Receiver>& receiver;

                [[noreturn]] void operator()(pika::detail::monostate) const { PIKA_UNREACHABLE; }

                void operator()(pika::execution::detail::stopped_type)
                {
                    pika::execution::experimental::set_stopped(std::move(receiver));
                }

                void operator()(error_type const& error)
                {
                    pika::detail::visit(error_visitor<Receiver>{receiver}, error);
                }

                void operator()(value_type const& ts)
                {
                    pika::detail::visit(value_visitor<Receiver>{receiver}, ts);
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
                    std::lock_guard<mutex_type> l{mtx};
                }

                if (!continuations.empty())
                {
                    for (auto const& continuation : continuations) { continuation(); }
                    continuations.clear();
                }
            }

            template <typename Receiver>
            void add_continuation(Receiver& receiver)
            {
                if (predecessor_done)
                {
                    // If we read predecessor_done here it means that one of
                    // set_error/set_stopped/set_value has been called and
                    // values/errors have been stored into the shared state.
                    // We can trigger the continuation directly.
                    // TODO: Should this preserve the scheduler? It does not
                    // if we call set_* inline.
                    pika::detail::visit(stopped_error_value_visitor<Receiver>{receiver}, v);
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
                        pika::detail::visit(stopped_error_value_visitor<Receiver>{receiver}, v);
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
                        continuations.emplace_back([this, &receiver]() mutable {
                            pika::detail::visit(stopped_error_value_visitor<Receiver>{receiver}, v);
                        });
                    }
                }
            }

            void start() & noexcept
            {
                if (!start_called.exchange(true))
                {
                    PIKA_ASSERT(os.has_value());
                    pika::execution::experimental::start(*os);
                }
            }

            friend void intrusive_ptr_add_ref(shared_state* p) { ++p->reference_count; }

            friend void intrusive_ptr_release(shared_state* p)
            {
                if (--p->reference_count == 0)
                {
                    allocator_type other_alloc(p->alloc);
                    std::allocator_traits<allocator_type>::destroy(other_alloc, p);
                    std::allocator_traits<allocator_type>::deallocate(other_alloc, p, 1);
                }
            }
        };

        pika::intrusive_ptr<shared_state> state;

        template <typename Sender_>
        split_sender_type(Sender_&& sender, Allocator const& allocator)
        {
            using allocator_type = Allocator;
            using other_allocator =
                typename std::allocator_traits<allocator_type>::template rebind_alloc<shared_state>;
            using allocator_traits = std::allocator_traits<other_allocator>;
            using unique_ptr =
                std::unique_ptr<shared_state, pika::detail::allocator_deleter<other_allocator>>;

            other_allocator alloc(allocator);
            unique_ptr p(allocator_traits::allocate(alloc, 1),
                pika::detail::allocator_deleter<other_allocator>{alloc});

            new (p.get()) shared_state{std::forward<Sender_>(sender), allocator};
            state = p.release();
        }

        split_sender_type(split_sender_type const&) = default;
        split_sender_type& operator=(split_sender_type const&) = default;
        split_sender_type(split_sender_type&&) = default;
        split_sender_type& operator=(split_sender_type&&) = default;

        template <typename Receiver>
        struct operation_state
        {
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            pika::intrusive_ptr<shared_state> state;

            template <typename Receiver_>
            operation_state(Receiver_&& receiver, pika::intrusive_ptr<shared_state> state)
              : receiver(std::forward<Receiver_>(receiver))
              , state(std::move(state))
            {
            }

            operation_state(operation_state&&) = delete;
            operation_state& operator=(operation_state&&) = delete;
            operation_state(operation_state const&) = delete;
            operation_state& operator=(operation_state const&) = delete;

            void start() & noexcept
            {
                state->start();
                state->add_continuation(receiver);
            }
        };

        template <typename Receiver>
        friend operation_state<Receiver> tag_invoke(
            pika::execution::experimental::connect_t, split_sender_type&& s, Receiver&& receiver)
        {
            return {std::forward<Receiver>(receiver), std::move(s.state)};
        }

        template <typename Receiver>
        friend operation_state<Receiver> tag_invoke(pika::execution::experimental::connect_t,
            split_sender_type const& s, Receiver&& receiver)
        {
            return {std::forward<Receiver>(receiver), s.state};
        }
    };

    template <typename Sender, typename Enable = void>
    struct is_split_sender_impl : std::false_type
    {
    };

    template <typename Sender>
    struct is_split_sender_impl<Sender, std::void_t<typename Sender::split_sender_tag>>
      : std::true_type
    {
    };

    template <typename Sender>
    inline constexpr bool is_split_sender_v = is_split_sender_impl<std::decay_t<Sender>>::value;
}    // namespace pika::split_detail

namespace pika::execution::experimental {
    inline constexpr struct split_t final : pika::functional::detail::tag_fallback<split_t>
    {
    private:
        template <typename Sender,
            PIKA_CONCEPT_REQUIRES_(is_sender_v<Sender> && !split_detail::is_split_sender_v<Sender>)>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(split_t, Sender&& sender)
        {
            return split_detail::split_sender<Sender, pika::detail::internal_allocator<>>{
                std::forward<Sender>(sender), {}};
        }

        template <typename Sender, typename Allocator,
            PIKA_CONCEPT_REQUIRES_(is_sender_v<Sender> &&
                !split_detail::is_split_sender_v<Sender> &&
                pika::detail::is_allocator_v<Allocator>)>
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(split_t, Sender&& sender, Allocator const& allocator)
        {
            return split_detail::split_sender<Sender, Allocator>{
                std::forward<Sender>(sender), allocator};
        }

        template <typename Sender, typename Allocator,
            PIKA_CONCEPT_REQUIRES_(split_detail::is_split_sender_v<Sender>&&
                    std::is_same_v<typename Sender::allocator_type, std::decay_t<Allocator>>)>
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(split_t, Sender&& sender, Allocator const&)
        {
            return std::forward<Sender>(sender);
        }

        template <typename Sender, typename Allocator,
            PIKA_CONCEPT_REQUIRES_(split_detail::is_split_sender_v<Sender> &&
                !std::is_same_v<typename Sender::allocator_type, std::decay_t<Allocator>>)>
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(split_t, Sender&& sender, Allocator const& allocator)
        {
            return split_detail::split_sender<Sender, Allocator>{
                std::forward<Sender>(sender), allocator};
        }

        template <typename Sender, PIKA_CONCEPT_REQUIRES_(split_detail::is_split_sender_v<Sender>)>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(split_t, Sender&& sender)
        {
            return std::forward<Sender>(sender);
        }

        template <typename Allocator = pika::detail::internal_allocator<>,
            PIKA_CONCEPT_REQUIRES_(pika::detail::is_allocator_v<Allocator>)>
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(split_t, Allocator const& allocator = {})
        {
            return detail::partial_algorithm<split_t, Allocator>{allocator};
        }
    } split{};
}    // namespace pika::execution::experimental
#endif
