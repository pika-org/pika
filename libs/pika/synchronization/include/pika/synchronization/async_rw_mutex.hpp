//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/assert.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/allocator_support/internal_allocator.hpp>
#include <pika/concurrency/spinlock.hpp>
#include <pika/datastructures/detail/small_vector.hpp>
#include <pika/execution_base/operation_state.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/execution_base/this_thread.hpp>
#include <pika/functional/unique_function.hpp>

#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>
#endif

namespace pika::execution::experimental {
    /// The type of access provided by async_rw_mutex.
    enum class async_rw_mutex_access_type
    {
        read,
        readwrite
    };

    namespace detail {
        template <typename T>
        struct async_rw_mutex_shared_state
        {
            using mutex_type = pika::concurrency::detail::spinlock;
            using shared_state_ptr_type = std::shared_ptr<async_rw_mutex_shared_state>;
            std::atomic<bool> value_set{false};
            std::optional<T> value{std::nullopt};
            shared_state_ptr_type next_state{nullptr};
            mutex_type mtx{};
            pika::detail::small_vector<
                pika::util::detail::unique_function<void(shared_state_ptr_type)>, 1>
                continuations{};

            async_rw_mutex_shared_state() = default;
            async_rw_mutex_shared_state(async_rw_mutex_shared_state&&) = delete;
            async_rw_mutex_shared_state& operator=(async_rw_mutex_shared_state&&) = delete;
            async_rw_mutex_shared_state(async_rw_mutex_shared_state const&) = delete;
            async_rw_mutex_shared_state& operator=(async_rw_mutex_shared_state const&) = delete;

            ~async_rw_mutex_shared_state()
            {
                // If there is no next state the continuations must be empty.
                PIKA_ASSERT(next_state || continuations.empty());

                // This state must always have the value set by the time it is
                // destructed. If there is no next state the value is destructed
                // with this state.
                PIKA_ASSERT(value);

                if (PIKA_LIKELY(next_state))
                {
                    // The current state has now finished all accesses to the
                    // wrapped value, so we move the value to the next state.
                    PIKA_ASSERT(value.has_value());
                    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                    next_state->set_value(PIKA_MOVE(*value));

                    for (auto& continuation : continuations) { continuation(next_state); }
                }
            }

            template <typename U>
            void set_value(U&& u)
            {
                PIKA_ASSERT(!value);
                value.emplace(PIKA_FORWARD(U, u));
                value_set.store(true, std::memory_order_release);
            }

            T& get_value()
            {
                pika::util::yield_while(
                    [this]() { return !value_set.load(std::memory_order_acquire); });
                PIKA_ASSERT(value);
                // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                return *value;
            }

            void set_next_state(std::shared_ptr<async_rw_mutex_shared_state> state)
            {
                // The next state should only be set once
                PIKA_ASSERT(!next_state);
                PIKA_ASSERT(state);
                next_state = PIKA_MOVE(state);
            }

            template <typename F>
            void add_continuation(F&& continuation)
            {
                std::lock_guard<mutex_type> l(mtx);
                continuations.emplace_back(PIKA_FORWARD(F, continuation));
            }
        };

        template <>
        struct async_rw_mutex_shared_state<void>
        {
            using mutex_type = pika::concurrency::detail::spinlock;
            using shared_state_ptr_type = std::shared_ptr<async_rw_mutex_shared_state>;
            shared_state_ptr_type next_state{nullptr};
            mutex_type mtx{};
            pika::detail::small_vector<
                pika::util::detail::unique_function<void(shared_state_ptr_type)>, 1>
                continuations{};

            async_rw_mutex_shared_state() = default;
            async_rw_mutex_shared_state(async_rw_mutex_shared_state&&) = delete;
            async_rw_mutex_shared_state& operator=(async_rw_mutex_shared_state&&) = delete;
            async_rw_mutex_shared_state(async_rw_mutex_shared_state const&) = delete;
            async_rw_mutex_shared_state& operator=(async_rw_mutex_shared_state const&) = delete;

            ~async_rw_mutex_shared_state()
            {
                // If there is no next state the continuations must be empty.
                PIKA_ASSERT(next_state || continuations.empty());

                for (auto& continuation : continuations) { continuation(next_state); }
            }

            void set_next_state(std::shared_ptr<async_rw_mutex_shared_state> state)
            {
                // The next state should only be set once
                PIKA_ASSERT(!next_state);
                PIKA_ASSERT(state);
                next_state = PIKA_MOVE(state);
            }

            template <typename F>
            void add_continuation(F&& continuation)
            {
                std::lock_guard<mutex_type> l(mtx);
                continuations.emplace_back(PIKA_FORWARD(F, continuation));
            }
        };
    }    // namespace detail

    /// A wrapper for values sent by senders from async_rw_mutex.
    ///
    /// All values sent by async_rw_mutex::read and async_rw_mutex::readwrite
    /// are wrapped by this class. It acts as a lock on the wrapped object and
    /// manages the lifetime of it. The wrapper has reference semantics. When
    /// the access type is readwrite the wrapper is only movable. When the last
    /// copy of a wrapper is released the next access through the async_rw_mutex
    /// (if any) will be triggered.
    template <typename ReadWriteT, typename ReadT, async_rw_mutex_access_type AccessType>
    struct async_rw_mutex_access_wrapper;

    template <typename ReadWriteT, typename ReadT>
    struct async_rw_mutex_access_wrapper<ReadWriteT, ReadT, async_rw_mutex_access_type::read>
    {
    private:
        using shared_state_type = std::shared_ptr<detail::async_rw_mutex_shared_state<ReadWriteT>>;
        shared_state_type state;

    public:
        async_rw_mutex_access_wrapper() = delete;
        async_rw_mutex_access_wrapper(shared_state_type state)
          : state(PIKA_MOVE(state))
        {
        }
        async_rw_mutex_access_wrapper(async_rw_mutex_access_wrapper&&) = default;
        async_rw_mutex_access_wrapper& operator=(async_rw_mutex_access_wrapper&&) = default;
        async_rw_mutex_access_wrapper(async_rw_mutex_access_wrapper const&) = default;
        async_rw_mutex_access_wrapper& operator=(async_rw_mutex_access_wrapper const&) = default;

        ReadT& get() const
        {
            PIKA_ASSERT(state);
            return state->get_value();
        }
    };

    template <typename ReadWriteT, typename ReadT>
    struct async_rw_mutex_access_wrapper<ReadWriteT, ReadT, async_rw_mutex_access_type::readwrite>
    {
    private:
        static_assert(!std::is_void<ReadWriteT>::value,
            "Cannot mix void and non-void type in async_rw_mutex_access_wrapper wrapper "
            "(ReadWriteT is void, ReadT is non-void)");
        static_assert(!std::is_void<ReadT>::value,
            "Cannot mix void and non-void type in async_rw_mutex_access_wrapper wrapper (ReadT "
            "is void, ReadWriteT is non-void)");

        using shared_state_type = std::shared_ptr<detail::async_rw_mutex_shared_state<ReadWriteT>>;
        shared_state_type state;

    public:
        async_rw_mutex_access_wrapper() = delete;
        async_rw_mutex_access_wrapper(shared_state_type state)
          : state(PIKA_MOVE(state))
        {
        }
        async_rw_mutex_access_wrapper(async_rw_mutex_access_wrapper&&) = default;
        async_rw_mutex_access_wrapper& operator=(async_rw_mutex_access_wrapper&&) = default;
        async_rw_mutex_access_wrapper(async_rw_mutex_access_wrapper const&) = delete;
        async_rw_mutex_access_wrapper& operator=(async_rw_mutex_access_wrapper const&) = delete;

        ReadWriteT& get()
        {
            PIKA_ASSERT(state);
            return state->get_value();
        }
    };

    // The void wrappers for read and readwrite are identical, but must be
    // specialized separately to avoid ambiguity with the non-void
    // specializations above.
    template <>
    struct async_rw_mutex_access_wrapper<void, void, async_rw_mutex_access_type::read>
    {
    private:
        using shared_state_type = std::shared_ptr<detail::async_rw_mutex_shared_state<void>>;
        shared_state_type state;

    public:
        async_rw_mutex_access_wrapper() = delete;
        explicit async_rw_mutex_access_wrapper(shared_state_type state)
          : state(PIKA_MOVE(state))
        {
        }
        async_rw_mutex_access_wrapper(async_rw_mutex_access_wrapper&&) = default;
        async_rw_mutex_access_wrapper& operator=(async_rw_mutex_access_wrapper&&) = default;
        async_rw_mutex_access_wrapper(async_rw_mutex_access_wrapper const&) = default;
        async_rw_mutex_access_wrapper& operator=(async_rw_mutex_access_wrapper const&) = default;
    };

    template <>
    struct async_rw_mutex_access_wrapper<void, void, async_rw_mutex_access_type::readwrite>
    {
    private:
        using shared_state_type = std::shared_ptr<detail::async_rw_mutex_shared_state<void>>;
        shared_state_type state;

    public:
        async_rw_mutex_access_wrapper() = delete;
        explicit async_rw_mutex_access_wrapper(shared_state_type state)
          : state(PIKA_MOVE(state))
        {
        }
        async_rw_mutex_access_wrapper(async_rw_mutex_access_wrapper&&) = default;
        async_rw_mutex_access_wrapper& operator=(async_rw_mutex_access_wrapper&&) = default;
        async_rw_mutex_access_wrapper(async_rw_mutex_access_wrapper const&) = delete;
        async_rw_mutex_access_wrapper& operator=(async_rw_mutex_access_wrapper const&) = delete;
    };

    /// Read-write mutex where access is granted to a value through senders.
    ///
    /// The wrapped value is accessed through read and readwrite, both of which
    /// return senders which call set_value on a connected receiver when the
    /// wrapped value is safe to read or write. The senders send the value
    /// through a wrapper type which is implicitly convertible to a reference of
    /// the wrapped value. Read-only senders send wrappers that are convertible
    /// to const references.
    ///
    /// A read-write sender gives exclusive access to the wrapped value, while a
    /// read-only sender gives shared (with other read-only senders) access to
    /// the value.
    ///
    /// A void mutex acts as a mutex around some user-managed resource, i.e. the
    /// void mutex does not manage any value and the types sent by the senders
    /// are not convertible. The sent types are copyable and release access to
    /// the protected resource when released.
    ///
    /// The order in which senders call set_value is determined by the order in
    /// which the senders are retrieved from the mutex. Connecting and starting
    /// the senders is thread-safe.
    ///
    /// Retrieving senders from the mutex is not thread-safe.
    ///
    /// The mutex is movable and non-copyable.
    template <typename ReadWriteT = void, typename ReadT = ReadWriteT,
        typename Allocator = pika::detail::internal_allocator<>>
    class async_rw_mutex;

    // Implementation details:
    //
    // The async_rw_mutex protects access to a given resource using two
    // reference counted shared states, the current and the previous state. Each
    // shared state guards access to the next stage; when the shared state goes
    // out of scope it triggers continuations for the next stage.
    //
    // When read-write access is required a sender is created which holds on to
    // the newly created shared state for the read-write access and the previous
    // state. When the sender is connected to a receiver, a callback is added to
    // the previous shared state's destructor. The callback holds the new state,
    // and passes a wrapper holding the shared state to set_value. Once the
    // receiver which receives the wrapper has let the wrapper go out of scope
    // (and all other references to the shared state are out of scope), the new
    // shared state will again trigger its continuations.
    //
    // When read-only access is required and the previous access was read-only
    // the procedure is the same as for read-write access. When read-only access
    // follows a previous read-only access the shared state is reused between
    // all consecutive read-only accesses, such that multiple read-only accesses
    // can run concurrently, and the next access (which must be read-write) is
    // triggered once all instances of that shared state have gone out of scope.
    //
    // The protected value is moved from state to state and is released when the
    // last shared state is destroyed.

    template <typename Allocator>
    class async_rw_mutex<void, void, Allocator>
    {
    private:
        template <async_rw_mutex_access_type AccessType>
        struct sender;

        using shared_state_type = detail::async_rw_mutex_shared_state<void>;
        using shared_state_weak_ptr_type = std::weak_ptr<shared_state_type>;

        // nvc++ is not able to see this typedef unless it's public
#if defined(PIKA_NVHPC_VERSION)
    public:
#endif
        using shared_state_ptr_type = std::shared_ptr<shared_state_type>;

    public:
        using read_type = void;
        using readwrite_type = void;

        using read_access_type = async_rw_mutex_access_wrapper<readwrite_type, read_type,
            async_rw_mutex_access_type::read>;
        using readwrite_access_type = async_rw_mutex_access_wrapper<readwrite_type, read_type,
            async_rw_mutex_access_type::readwrite>;

        using allocator_type = Allocator;

        explicit async_rw_mutex(allocator_type const& alloc = {})
          : alloc(alloc)
        {
        }
        async_rw_mutex(async_rw_mutex&&) = default;
        async_rw_mutex& operator=(async_rw_mutex&&) = default;
        async_rw_mutex(async_rw_mutex const&) = delete;
        async_rw_mutex& operator=(async_rw_mutex const&) = delete;

        sender<async_rw_mutex_access_type::read> read()
        {
            if (prev_access == async_rw_mutex_access_type::readwrite)
            {
                auto shared_prev_state = PIKA_MOVE(state);
                state = std::allocate_shared<shared_state_type, allocator_type>(alloc);
                prev_access = async_rw_mutex_access_type::read;

                // Only the first access has no previous shared state. When
                // there is a previous state we set the next state so that the
                // value can be passed from the previous state to the next
                // state.
                if (PIKA_LIKELY(shared_prev_state))
                {
                    shared_prev_state->set_next_state(state);
                    prev_state = shared_prev_state;
                }
            }

            return {prev_state, state};
        }

        sender<async_rw_mutex_access_type::readwrite> readwrite()
        {
            auto shared_prev_state = PIKA_MOVE(state);
            state = std::allocate_shared<shared_state_type, allocator_type>(alloc);
            prev_access = async_rw_mutex_access_type::readwrite;

            // Only the first access has no previous shared state. When there is
            // a previous state we set the next state so that the value can be
            // passed from the previous state to the next state.
            if (PIKA_LIKELY(shared_prev_state))
            {
                shared_prev_state->set_next_state(state);
                prev_state = shared_prev_state;
            }

            return {prev_state, state};
        }

    private:
        template <async_rw_mutex_access_type AccessType>
        struct sender
        {
            PIKA_STDEXEC_SENDER_CONCEPT

            shared_state_weak_ptr_type prev_state;
            shared_state_ptr_type state;

            using access_type =
                async_rw_mutex_access_wrapper<readwrite_type, read_type, AccessType>;
            template <template <typename...> class Tuple, template <typename...> class Variant>
            using value_types = Variant<Tuple<access_type>>;

            template <template <typename...> class Variant>
            using error_types = Variant<std::exception_ptr>;

            static constexpr bool sends_done = false;

            using completion_signatures = pika::execution::experimental::completion_signatures<
                pika::execution::experimental::set_value_t(access_type),
                pika::execution::experimental::set_error_t(std::exception_ptr)>;

            template <typename R>
            struct operation_state
            {
                std::decay_t<R> r;
                shared_state_weak_ptr_type prev_state;
                shared_state_ptr_type state;

                template <typename R_>
                operation_state(
                    R_&& r, shared_state_weak_ptr_type prev_state, shared_state_ptr_type state)
                  : r(PIKA_FORWARD(R_, r))
                  , prev_state(PIKA_MOVE(prev_state))
                  , state(PIKA_MOVE(state))
                {
                }

                operation_state(operation_state&&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                friend void tag_invoke(
                    pika::execution::experimental::start_t, operation_state& os) noexcept
                {
                    PIKA_ASSERT_MSG(os.state,
                        "async_rw_lock::sender::operation_state state is empty, was the sender "
                        "already started?");

                    auto continuation = [r = PIKA_MOVE(os.r)](shared_state_ptr_type state) mutable {
                        try
                        {
                            pika::execution::experimental::set_value(
                                PIKA_MOVE(r), access_type{PIKA_MOVE(state)});
                        }
                        catch (...)
                        {
                            pika::execution::experimental::set_error(
                                PIKA_MOVE(r), std::current_exception());
                        }
                    };

                    if (auto p = os.prev_state.lock())
                    {
                        // If the previous state is set and it's still alive,
                        // add a continuation to be triggered when the previous
                        // state is released.
                        p->add_continuation(PIKA_MOVE(continuation));
                        os.state.reset();
                        os.prev_state.reset();
                    }
                    else
                    {
                        // There is no previous state on the first access or the
                        // previous state has already been released. We can run
                        // the continuation immediately.
                        continuation(PIKA_MOVE(os.state));
                    }
                }
            };

            template <typename R>
            friend auto tag_invoke(pika::execution::experimental::connect_t, sender&& s, R&& r)
            {
                return operation_state<R>{
                    PIKA_FORWARD(R, r), PIKA_MOVE(s.prev_state), PIKA_MOVE(s.state)};
            }
        };

        PIKA_NO_UNIQUE_ADDRESS allocator_type alloc;

        async_rw_mutex_access_type prev_access = async_rw_mutex_access_type::readwrite;

        shared_state_weak_ptr_type prev_state;
        shared_state_ptr_type state;
    };

    template <typename ReadWriteT, typename ReadT, typename Allocator>
    class async_rw_mutex
    {
    private:
        static_assert(!std::is_void<ReadWriteT>::value,
            "Cannot mix void and non-void type in async_rw_mutex (ReadWriteT is void, ReadT is "
            "non-void)");
        static_assert(!std::is_void<ReadT>::value,
            "Cannot mix void and non-void type in async_rw_mutex (ReadT is void, ReadWriteT is "
            "non-void)");

        template <async_rw_mutex_access_type AccessType>
        struct sender;

    public:
        using read_type = std::decay_t<ReadT> const;
        using readwrite_type = std::decay_t<ReadWriteT>;
        using value_type = readwrite_type;

        using read_access_type = async_rw_mutex_access_wrapper<readwrite_type, read_type,
            async_rw_mutex_access_type::read>;
        using readwrite_access_type = async_rw_mutex_access_wrapper<readwrite_type, read_type,
            async_rw_mutex_access_type::readwrite>;

        using allocator_type = Allocator;

        async_rw_mutex() = delete;
        template <typename U,
            typename = std::enable_if_t<!std::is_same<std::decay_t<U>, async_rw_mutex>::value>>
        explicit async_rw_mutex(U&& u, allocator_type const& alloc = {})
          : value(PIKA_FORWARD(U, u))
          , alloc(alloc)
        {
        }
        async_rw_mutex(async_rw_mutex&&) = default;
        async_rw_mutex& operator=(async_rw_mutex&&) = default;
        async_rw_mutex(async_rw_mutex const&) = delete;
        async_rw_mutex& operator=(async_rw_mutex const&) = delete;

        sender<async_rw_mutex_access_type::read> read()
        {
            if (prev_access == async_rw_mutex_access_type::readwrite)
            {
                auto shared_prev_state = PIKA_MOVE(state);
                state = std::allocate_shared<shared_state_type, allocator_type>(alloc);
                prev_access = async_rw_mutex_access_type::read;

                // Only the first access has no previous shared state. When
                // there is a previous state we set the next state so that the
                // value can be passed from the previous state to the next
                // state. When there is no previous state we need to move the
                // value to the first state.
                if (PIKA_LIKELY(shared_prev_state))
                {
                    shared_prev_state->set_next_state(state);
                    prev_state = shared_prev_state;
                }
                else { state->set_value(PIKA_MOVE(value)); }
            }

            return {prev_state, state};
        }

        sender<async_rw_mutex_access_type::readwrite> readwrite()
        {
            auto shared_prev_state = PIKA_MOVE(state);
            state = std::allocate_shared<shared_state_type, allocator_type>(alloc);
            prev_access = async_rw_mutex_access_type::readwrite;

            // Only the first access has no previous shared state. When there is
            // a previous state we set the next state so that the value can be
            // passed from the previous state to the next state. When there is
            // no previous state we need to move the value to the first state.
            if (PIKA_LIKELY(shared_prev_state))
            {
                shared_prev_state->set_next_state(state);
                prev_state = shared_prev_state;
            }
            else { state->set_value(PIKA_MOVE(value)); }

            return {prev_state, state};
        }

    private:
        using shared_state_type = detail::async_rw_mutex_shared_state<value_type>;
        using shared_state_weak_ptr_type = std::weak_ptr<shared_state_type>;

        // nvc++ is not able to see this typedef unless it's public
#if defined(PIKA_NVHPC_VERSION)
    public:
#endif
        using shared_state_ptr_type = std::shared_ptr<shared_state_type>;

    private:
        template <async_rw_mutex_access_type AccessType>
        struct sender
        {
            PIKA_STDEXEC_SENDER_CONCEPT

            shared_state_weak_ptr_type prev_state;
            shared_state_ptr_type state;

            using access_type =
                async_rw_mutex_access_wrapper<readwrite_type, read_type, AccessType>;
            template <template <typename...> class Tuple, template <typename...> class Variant>
            using value_types = Variant<Tuple<access_type>>;

            template <template <typename...> class Variant>
            using error_types = Variant<std::exception_ptr>;

            static constexpr bool sends_done = false;

            using completion_signatures = pika::execution::experimental::completion_signatures<
                pika::execution::experimental::set_value_t(access_type),
                pika::execution::experimental::set_error_t(std::exception_ptr)>;

            template <typename R>
            struct operation_state
            {
                std::decay_t<R> r;
                shared_state_weak_ptr_type prev_state;
                shared_state_ptr_type state;

                template <typename R_>
                operation_state(
                    R_&& r, shared_state_weak_ptr_type prev_state, shared_state_ptr_type state)
                  : r(PIKA_FORWARD(R_, r))
                  , prev_state(PIKA_MOVE(prev_state))
                  , state(PIKA_MOVE(state))
                {
                }

                operation_state(operation_state&&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                friend void tag_invoke(
                    pika::execution::experimental::start_t, operation_state& os) noexcept
                {
                    PIKA_ASSERT_MSG(os.state,
                        "async_rw_lock::sender::operation_state state is empty, was the sender "
                        "already started?");

                    auto continuation = [r = PIKA_MOVE(os.r)](shared_state_ptr_type state) mutable {
                        try
                        {
                            pika::execution::experimental::set_value(
                                PIKA_MOVE(r), access_type{PIKA_MOVE(state)});
                        }
                        catch (...)
                        {
                            pika::execution::experimental::set_error(
                                PIKA_MOVE(r), std::current_exception());
                        }
                    };

                    if (auto p = os.prev_state.lock())
                    {
                        // If the previous state is set and it's still alive,
                        // add a continuation to be triggered when the previous
                        // state is released.
                        p->add_continuation(PIKA_MOVE(continuation));
                        os.state.reset();
                        os.prev_state.reset();
                    }
                    else
                    {
                        // There is no previous state on the first access or the
                        // previous state has already been released. We can run
                        // the continuation immediately.
                        continuation(PIKA_MOVE(os.state));
                    }
                }
            };

            template <typename R>
            friend auto tag_invoke(pika::execution::experimental::connect_t, sender&& s, R&& r)
            {
                return operation_state<R>{
                    PIKA_FORWARD(R, r), PIKA_MOVE(s.prev_state), PIKA_MOVE(s.state)};
            }

            template <typename R>
            friend auto tag_invoke(pika::execution::experimental::connect_t, sender const& s, R&& r)
            {
                if constexpr (AccessType == async_rw_mutex_access_type::readwrite)
                {
                    static_assert(sizeof(R) == 0,
                        "senders returned from async_rw_mutex::readwrite are not l-lvalue "
                        "connectable");
                }

                return operation_state<R>{PIKA_FORWARD(R, r), s.prev_state, s.state};
            }
        };

        PIKA_NO_UNIQUE_ADDRESS value_type value;
        PIKA_NO_UNIQUE_ADDRESS allocator_type alloc;

        async_rw_mutex_access_type prev_access = async_rw_mutex_access_type::readwrite;

        shared_state_weak_ptr_type prev_state;
        shared_state_ptr_type state;
    };
}    // namespace pika::execution::experimental
