//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/allocator_support/allocator_deleter.hpp>
#include <pika/allocator_support/internal_allocator.hpp>
#include <pika/allocator_support/traits/is_allocator.hpp>
#include <pika/assert.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/datastructures/detail/small_vector.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/datastructures/variant.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution/algorithms/detail/single_result.hpp>
#include <pika/execution_base/operation_state.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/bind_front.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/invoke_fused.hpp>
#include <pika/functional/unique_function.hpp>
#include <pika/modules/memory.hpp>
#include <pika/synchronization/spinlock.hpp>
#include <pika/thread_support/atomic_count.hpp>
#include <pika/type_support/pack.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

namespace pika { namespace execution { namespace experimental {
    namespace detail {
        enum class submission_type
        {
            eager,
            lazy
        };

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

        template <typename Receiver>
        struct value_visitor
        {
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

            template <typename Ts>
            void operator()(Ts const& ts)
            {
                pika::util::invoke_fused(
                    pika::util::bind_front(
                        pika::execution::experimental::set_value,
                        PIKA_MOVE(receiver)),
                    ts);
            }
        };

        template <typename Sender, typename Allocator, submission_type Type>
        struct split_sender
        {
            template <typename Tuple>
            struct value_types_helper
            {
                using const_type =
                    pika::util::detail::transform_t<Tuple, std::add_const>;
                using type = pika::util::detail::transform_t<const_type,
                    std::add_lvalue_reference>;
            };

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types = pika::util::detail::transform_t<
                typename pika::execution::experimental::sender_traits<
                    Sender>::template value_types<Tuple, Variant>,
                value_types_helper>;

            template <template <typename...> class Variant>
            using error_types =
                pika::util::detail::unique_t<pika::util::detail::prepend_t<
                    typename pika::execution::experimental::sender_traits<
                        Sender>::template error_types<Variant>,
                    std::exception_ptr>>;

            static constexpr bool sends_done = false;

            struct shared_state
            {
                struct split_receiver;

                using allocator_type = typename std::allocator_traits<
                    Allocator>::template rebind_alloc<shared_state>;
                PIKA_NO_UNIQUE_ADDRESS allocator_type alloc;
                using mutex_type = pika::lcos::local::spinlock;
                mutex_type mtx;
                pika::util::atomic_count reference_count{0};
                std::atomic<bool> start_called{false};
                std::atomic<bool> predecessor_done{false};

                using operation_state_type =
                    std::decay_t<connect_result_t<Sender, split_receiver>>;
                operation_state_type os;

                struct done_type
                {
                };
                using value_type =
                    typename pika::execution::experimental::sender_traits<
                        Sender>::template value_types<pika::tuple, pika::variant>;
                using error_type =
                    pika::util::detail::unique_t<pika::util::detail::prepend_t<
                        error_types<pika::variant>, std::exception_ptr>>;
                pika::variant<pika::monostate, done_type, error_type, value_type>
                    v;

                using continuation_type =
                    pika::util::unique_function_nonser<void()>;
                pika::detail::small_vector<continuation_type, 1> continuations;

                struct split_receiver
                {
                    pika::intrusive_ptr<shared_state> state;

                    template <typename Error>
                    friend void tag_invoke(
                        set_error_t, split_receiver&& r, Error&& error) noexcept
                    {
                        r.state->v.template emplace<error_type>(
                            error_type(PIKA_FORWARD(Error, error)));
                        r.state->set_predecessor_done();
                        r.state.reset();
                    }

                    friend void tag_invoke(
                        set_done_t, split_receiver&& r) noexcept
                    {
                        r.state->set_predecessor_done();
                        r.state.reset();
                    };

                    // This typedef is duplicated from the parent struct. The
                    // parent typedef is not instantiated early enough for use
                    // here.
                    using value_type =
                        typename pika::execution::experimental::sender_traits<
                            Sender>::template value_types<pika::tuple,
                            pika::variant>;

                    template <typename... Ts>
                    friend auto tag_invoke(
                        set_value_t, split_receiver&& r, Ts&&... ts) noexcept
                        -> decltype(
                            std::declval<
                                pika::variant<pika::monostate, value_type>>()
                                .template emplace<value_type>(
                                    pika::make_tuple<>(PIKA_FORWARD(Ts, ts)...)),
                            void())
                    {
                        r.state->v.template emplace<value_type>(
                            pika::make_tuple<>(PIKA_FORWARD(Ts, ts)...));

                        r.state->set_predecessor_done();
                        r.state.reset();
                    }
                };

                template <typename Sender_,
                    typename = std::enable_if_t<!std::is_same<
                        std::decay_t<Sender_>, shared_state>::value>>
                shared_state(Sender_&& sender, allocator_type const& alloc)
                  : alloc(alloc)
                  , os(pika::execution::experimental::connect(
                        PIKA_FORWARD(Sender_, sender), split_receiver{this}))
                {
                }

                ~shared_state()
                {
                    PIKA_ASSERT_MSG(start_called,
                        "start was never called on the operation state of "
                        "split or ensure_started. Did you forget to connect the"
                        "sender to a receiver, or call start on the operation "
                        "state?");
                }

                template <typename Receiver>
                struct done_error_value_visitor
                {
                    PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

                    PIKA_NORETURN void operator()(pika::monostate) const
                    {
                        PIKA_UNREACHABLE;
                    }

                    void operator()(done_type)
                    {
                        pika::execution::experimental::set_done(
                            PIKA_MOVE(receiver));
                    }

                    void operator()(error_type const& error)
                    {
                        pika::visit(error_visitor<Receiver>{PIKA_FORWARD(
                                       Receiver, receiver)},
                            error);
                    }

                    void operator()(value_type const& ts)
                    {
                        pika::visit(value_visitor<Receiver>{PIKA_FORWARD(
                                       Receiver, receiver)},
                            ts);
                    }
                };

                void set_predecessor_done()
                {
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
                        for (auto const& continuation : continuations)
                        {
                            continuation();
                        }

                        continuations.clear();
                    }
                }

                template <typename Receiver>
                void add_continuation(Receiver& receiver) = delete;

                template <typename Receiver>
                void add_continuation(Receiver&& receiver)
                {
                    if (predecessor_done)
                    {
                        // If we read predecessor_done here it means that one of
                        // set_error/set_done/set_value has been called and
                        // values/errors have been stored into the shared state.
                        // We can trigger the continuation directly.
                        // TODO: Should this preserve the scheduler? It does not
                        // if we call set_* inline.
                        pika::visit(
                            done_error_value_visitor<Receiver>{
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
                            pika::visit(
                                done_error_value_visitor<Receiver>{
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
                            // when set_error/set_done/set_value is called.
                            continuations.emplace_back(
                                [this,
                                    receiver = PIKA_FORWARD(
                                        Receiver, receiver)]() mutable {
                                    pika::visit(
                                        done_error_value_visitor<Receiver>{
                                            PIKA_MOVE(receiver)},
                                        v);
                                });
                        }
                    }
                }

                void start() & noexcept
                {
                    if (!start_called.exchange(true))
                    {
                        pika::execution::experimental::start(os);
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
                        std::allocator_traits<allocator_type>::destroy(
                            other_alloc, p);
                        std::allocator_traits<allocator_type>::deallocate(
                            other_alloc, p, 1);
                    }
                }
            };

            pika::intrusive_ptr<shared_state> state;

            template <typename Sender_>
            split_sender(Sender_&& sender, Allocator const& allocator)
            {
                using allocator_type = Allocator;
                using other_allocator = typename std::allocator_traits<
                    allocator_type>::template rebind_alloc<shared_state>;
                using allocator_traits = std::allocator_traits<other_allocator>;
                using unique_ptr = std::unique_ptr<shared_state,
                    util::allocator_deleter<other_allocator>>;

                other_allocator alloc(allocator);
                unique_ptr p(allocator_traits::allocate(alloc, 1),
                    pika::util::allocator_deleter<other_allocator>{alloc});

                new (p.get())
                    shared_state{PIKA_FORWARD(Sender_, sender), allocator};
                state = p.release();

                // Eager submission means that we start the predecessor
                // operation state already when creating the sender. We don't
                // wait for another receiver to be connected.
                if constexpr (Type == submission_type::eager)
                {
                    state->start();
                }
            }

            split_sender(split_sender const&) = default;
            split_sender& operator=(split_sender const&) = default;
            split_sender(split_sender&&) = default;
            split_sender& operator=(split_sender&&) = default;

            template <typename Receiver>
            struct operation_state
            {
                PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
                pika::intrusive_ptr<shared_state> state;

                template <typename Receiver_>
                operation_state(Receiver_&& receiver,
                    pika::intrusive_ptr<shared_state> state)
                  : receiver(PIKA_FORWARD(Receiver_, receiver))
                  , state(PIKA_MOVE(state))
                {
                }

                operation_state(operation_state&&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                friend void tag_invoke(start_t, operation_state& os) noexcept
                {
                    // Lazy submission means that we wait to start the
                    // predecessor operation state when a downstream operation
                    // state is started, i.e. this start function is called.
                    if constexpr (Type == submission_type::lazy)
                    {
                        os.state->start();
                    }

                    os.state->add_continuation(PIKA_MOVE(os.receiver));
                }
            };

            template <typename Receiver>
            friend operation_state<Receiver> tag_invoke(
                connect_t, split_sender&& s, Receiver&& receiver)
            {
                return {PIKA_FORWARD(Receiver, receiver), PIKA_MOVE(s.state)};
            }

            template <typename Receiver>
            friend operation_state<Receiver> tag_invoke(
                connect_t, split_sender& s, Receiver&& receiver)
            {
                return {PIKA_FORWARD(Receiver, receiver), s.state};
            }
        };
    }    // namespace detail

    inline constexpr struct split_t final
      : pika::functional::detail::tag_fallback<split_t>
    {
    private:
        // clang-format off
        template <typename Sender,
            typename Allocator = pika::util::internal_allocator<>,
            PIKA_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                pika::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            split_t, Sender&& sender, Allocator const& allocator = {})
        {
            return detail::split_sender<Sender, Allocator,
                detail::submission_type::lazy>{
                PIKA_FORWARD(Sender, sender), allocator};
        }

        template <typename Sender, typename Allocator>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(split_t,
            detail::split_sender<Sender, Allocator,
                detail::submission_type::lazy>
                sender,
            Allocator const& = {})
        {
            return sender;
        }

        // clang-format off
        template <typename Allocator = pika::util::internal_allocator<>,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            split_t, Allocator const& allocator = {})
        {
            return detail::partial_algorithm<split_t, Allocator>{allocator};
        }
    } split{};
}}}    // namespace pika::execution::experimental
