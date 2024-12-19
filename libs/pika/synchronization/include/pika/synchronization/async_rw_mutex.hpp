//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/allocator_support/allocator_deleter.hpp>
#include <pika/allocator_support/internal_allocator.hpp>
#include <pika/assert.hpp>
#include <pika/datastructures/detail/small_vector.hpp>
#include <pika/execution_base/operation_state.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/execution_base/this_thread.hpp>
#include <pika/memory/intrusive_ptr.hpp>
#include <pika/thread_support/atomic_count.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>

namespace pika::execution::experimental {
    /// \brief The type of access provided by async_rw_mutex.
    enum class async_rw_mutex_access_type
    {
        /// \brief Read-only access.
        read,
        /// \brief Read-write access.
        readwrite
    };

    namespace detail {
        struct async_rw_mutex_operation_state_base
        {
            async_rw_mutex_operation_state_base* next{nullptr};
            virtual void continuation() = 0;
        };

        template <typename T, typename Allocator>
        struct async_rw_mutex_shared_state
        {
            using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<
                async_rw_mutex_shared_state>;
            using shared_state_ptr_type = pika::intrusive_ptr<async_rw_mutex_shared_state>;
            using value_ptr_type = std::shared_ptr<T>;

            PIKA_NO_UNIQUE_ADDRESS Allocator alloc;
            value_ptr_type value{nullptr};
            pika::detail::atomic_count reference_count{0};
            shared_state_ptr_type next_state{nullptr};
            std::atomic<void*> op_state_head{nullptr};

            async_rw_mutex_shared_state() = default;
            async_rw_mutex_shared_state(allocator_type&& alloc, value_ptr_type v)
              : alloc(std::move(alloc))
              , value(std::move(v))
            {
            }
            async_rw_mutex_shared_state(async_rw_mutex_shared_state&&) = delete;
            async_rw_mutex_shared_state& operator=(async_rw_mutex_shared_state&&) = delete;
            async_rw_mutex_shared_state(async_rw_mutex_shared_state const&) = delete;
            async_rw_mutex_shared_state& operator=(async_rw_mutex_shared_state const&) = delete;

            ~async_rw_mutex_shared_state()
            {
                if (next_state)
                {
                    // We pass the ownership of the intrusive_ptr of the next state to the next
                    // state itself, so that it can choose when to release it. If we can avoid it,
                    // we don't want this shared state to to hold on to the reference longer than
                    // necessary.
                    async_rw_mutex_shared_state* p = next_state.get();
                    p->done(std::move(next_state));
                }
            }

            void done(shared_state_ptr_type p) noexcept
            {
                while (true)
                {
                    void* expected = op_state_head.load(std::memory_order_relaxed);
                    // TODO: memory order
                    // this is not an async_rw_mutex_operation_state_base*, but is a known value to
                    // signal that the queue has been processed
                    if (op_state_head.compare_exchange_strong(expected, static_cast<void*>(this)))
                    {
                        // We have now successfully acquired the head of the queue, and signaled to
                        // other threads that they can't add any more items to the queue. We can now
                        // process the queue without further synchronization.
                        auto* current = static_cast<async_rw_mutex_operation_state_base*>(expected);

                        // We are also not accessing this shared state directly anymore, so we can
                        // reset p early.
                        p.reset();

                        while (current != nullptr)
                        {
                            auto* next = current->next;
                            current->continuation();
                            current = next;
                        }

                        break;
                    }
                }
            }

            void set_value(value_ptr_type v)
            {
                PIKA_ASSERT(v);
                PIKA_ASSERT(!value);
                value = std::move(v);
            }

            T& get_value()
            {
                PIKA_ASSERT(value);
                // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                return *value;
            }

            void set_next_state(shared_state_ptr_type state)
            {
                // The next state should only be set once
                PIKA_ASSERT(!next_state);
                PIKA_ASSERT(state);
                next_state = std::move(state);
            }

            bool add_op_state(async_rw_mutex_operation_state_base* op_state)
            {
                while (true)
                {
                    void* expected = op_state_head.load(std::memory_order_relaxed);
                    if (expected == static_cast<void*>(this)) { return false; }

                    op_state->next = static_cast<async_rw_mutex_operation_state_base*>(expected);
                    // TODO: memory order
                    if (op_state_head.compare_exchange_strong(
                            expected, static_cast<void*>(op_state)))
                    {
                        break;
                    }
                }

                return true;
            }

            friend void intrusive_ptr_add_ref(async_rw_mutex_shared_state* p)
            {
                ++p->reference_count;
            }

            friend void intrusive_ptr_release(async_rw_mutex_shared_state* p)
            {
                if (--p->reference_count == 0)
                {
                    allocator_type other_alloc(p->alloc);
                    std::allocator_traits<allocator_type>::destroy(other_alloc, p);
                    std::allocator_traits<allocator_type>::deallocate(other_alloc, p, 1);
                }
            }
        };

        template <typename Allocator>
        struct async_rw_mutex_shared_state<void, Allocator>
        {
            using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<
                async_rw_mutex_shared_state>;
            using shared_state_ptr_type = pika::intrusive_ptr<async_rw_mutex_shared_state>;

            PIKA_NO_UNIQUE_ADDRESS Allocator alloc;
            shared_state_ptr_type next_state{nullptr};
            pika::detail::atomic_count reference_count{0};
            std::atomic<void*> op_state_head{nullptr};

            async_rw_mutex_shared_state() = default;
            explicit async_rw_mutex_shared_state(allocator_type&& alloc)
              : alloc(std::move(alloc))
            {
            }
            async_rw_mutex_shared_state(async_rw_mutex_shared_state&&) = delete;
            async_rw_mutex_shared_state& operator=(async_rw_mutex_shared_state&&) = delete;
            async_rw_mutex_shared_state(async_rw_mutex_shared_state const&) = delete;
            async_rw_mutex_shared_state& operator=(async_rw_mutex_shared_state const&) = delete;

            ~async_rw_mutex_shared_state()
            {
                if (next_state)
                {
                    // We pass the ownership of the intrusive_ptr of the next state to the next
                    // state itself, so that it can choose when to release it. If we can avoid it,
                    // we don't want this shared state to to hold on to the reference longer than
                    // necessary.
                    async_rw_mutex_shared_state* p = next_state.get();
                    p->done(std::move(next_state));
                }
            }

            void done(shared_state_ptr_type p) noexcept
            {
                while (true)
                {
                    void* expected = op_state_head.load(std::memory_order_relaxed);
                    // TODO: memory order
                    // this is not an async_rw_mutex_operation_state_base*, but is a known value to
                    // signal that the queue has been processed
                    if (op_state_head.compare_exchange_strong(expected, static_cast<void*>(this)))
                    {
                        // We have now successfully acquired the head of the queue, and signaled to
                        // other threads that they can't add any more items to the queue. We can now
                        // process the queue without further synchronization.
                        auto* current = static_cast<async_rw_mutex_operation_state_base*>(expected);

                        // We are also not accessing this shared state directly anymore, so we can
                        // reset p early.
                        p.reset();

                        while (current != nullptr)
                        {
                            auto* next = current->next;
                            current->continuation();
                            current = next;
                        }

                        break;
                    }
                }
            }

            void set_next_state(shared_state_ptr_type state)
            {
                // The next state should only be set once
                PIKA_ASSERT(!next_state);
                PIKA_ASSERT(state);
                next_state = std::move(state);
            }

            bool add_op_state(async_rw_mutex_operation_state_base* op_state)
            {
                while (true)
                {
                    void* expected = op_state_head.load(std::memory_order_relaxed);
                    if (expected == static_cast<void*>(this)) { return false; }

                    op_state->next = static_cast<async_rw_mutex_operation_state_base*>(expected);
                    // TODO: memory order
                    if (op_state_head.compare_exchange_strong(
                            expected, static_cast<void*>(op_state)))
                    {
                        break;
                    }
                }

                return true;
            }

            friend void intrusive_ptr_add_ref(async_rw_mutex_shared_state* p)
            {
                ++p->reference_count;
            }

            friend void intrusive_ptr_release(async_rw_mutex_shared_state* p)
            {
                if (--p->reference_count == 0)
                {
                    allocator_type other_alloc(p->alloc);
                    std::allocator_traits<allocator_type>::destroy(other_alloc, p);
                    std::allocator_traits<allocator_type>::deallocate(other_alloc, p, 1);
                }
            }
        };
    }    // namespace detail

    /// \brief A wrapper for values sent by senders from \ref async_rw_mutex.
    ///
    /// All values sent by senders accessed through \ref async_rw_mutex are wrapped by this class.
    /// The wrapper has reference semantics to the wrapped object, and controls when subsequent
    /// accesses is given. When the destructor of the last or only wrapper runs, senders for
    /// subsequent accesses will signal their value channel.
    ///
    /// When the access type is \ref async_rw_mutex_access_type::readwrite the wrapper is move-only.
    /// When the access type is \ref async_rw_mutex_access_type::read the wrapper is copyable.
    template <typename ReadWriteT, typename ReadT, async_rw_mutex_access_type AccessType,
        typename Allocator>
    class async_rw_mutex_access_wrapper;

    /// \brief A wrapper for values sent by senders from \ref async_rw_mutex with read-only access.
    ///
    /// The wrapper is copyable.
    template <typename ReadWriteT, typename ReadT, typename Allocator>
    class async_rw_mutex_access_wrapper<ReadWriteT, ReadT, async_rw_mutex_access_type::read,
        Allocator>
    {
    private:
        using shared_state_ptr_type =
            pika::intrusive_ptr<detail::async_rw_mutex_shared_state<ReadWriteT, Allocator>>;
        shared_state_ptr_type state;

    public:
        async_rw_mutex_access_wrapper() = delete;
        explicit async_rw_mutex_access_wrapper(shared_state_ptr_type state)
          : state(std::move(state))
        {
        }
        async_rw_mutex_access_wrapper(async_rw_mutex_access_wrapper&&) = default;
        async_rw_mutex_access_wrapper& operator=(async_rw_mutex_access_wrapper&&) = default;
        async_rw_mutex_access_wrapper(async_rw_mutex_access_wrapper const&) = default;
        async_rw_mutex_access_wrapper& operator=(async_rw_mutex_access_wrapper const&) = default;

        /// \brief Access the wrapped type by const reference.
        ReadT& get() const
        {
            PIKA_ASSERT(state);
            return state->get_value();
        }
    };

    /// \brief A wrapper for values sent by senders from \ref async_rw_mutex with read-write access.
    ///
    /// The wrapper is move-only.
    template <typename ReadWriteT, typename ReadT, typename Allocator>
    class async_rw_mutex_access_wrapper<ReadWriteT, ReadT, async_rw_mutex_access_type::readwrite,
        Allocator>
    {
    private:
        static_assert(!std::is_void<ReadWriteT>::value,
            "Cannot mix void and non-void type in async_rw_mutex_access_wrapper wrapper "
            "(ReadWriteT is void, ReadT is non-void)");
        static_assert(!std::is_void<ReadT>::value,
            "Cannot mix void and non-void type in async_rw_mutex_access_wrapper wrapper (ReadT "
            "is void, ReadWriteT is non-void)");

        using shared_state_ptr_type =
            pika::intrusive_ptr<detail::async_rw_mutex_shared_state<ReadWriteT, Allocator>>;
        shared_state_ptr_type state;

    public:
        async_rw_mutex_access_wrapper() = delete;
        explicit async_rw_mutex_access_wrapper(shared_state_ptr_type state)
          : state(std::move(state))
        {
        }
        async_rw_mutex_access_wrapper(async_rw_mutex_access_wrapper&&) = default;
        async_rw_mutex_access_wrapper& operator=(async_rw_mutex_access_wrapper&&) = default;
        async_rw_mutex_access_wrapper(async_rw_mutex_access_wrapper const&) = delete;
        async_rw_mutex_access_wrapper& operator=(async_rw_mutex_access_wrapper const&) = delete;

        /// \brief Access the wrapped type by reference.
        ReadWriteT& get()
        {
            PIKA_ASSERT(state);
            return state->get_value();
        }
    };

    // The void wrappers for read and readwrite are identical, but must be
    // specialized separately to avoid ambiguity with the non-void
    // specializations above.

    /// \brief A wrapper for read-only access granted by a \p void \ref async_rw_mutex.
    ///
    /// The wrapper is copyable.
    template <typename Allocator>
    class async_rw_mutex_access_wrapper<void, void, async_rw_mutex_access_type::read, Allocator>
    {
    private:
        using shared_state_ptr_type =
            pika::intrusive_ptr<detail::async_rw_mutex_shared_state<void, Allocator>>;
        shared_state_ptr_type state;

    public:
        async_rw_mutex_access_wrapper() = delete;
        explicit async_rw_mutex_access_wrapper(shared_state_ptr_type state)
          : state(std::move(state))
        {
        }
        async_rw_mutex_access_wrapper(async_rw_mutex_access_wrapper&&) = default;
        async_rw_mutex_access_wrapper& operator=(async_rw_mutex_access_wrapper&&) = default;
        async_rw_mutex_access_wrapper(async_rw_mutex_access_wrapper const&) = default;
        async_rw_mutex_access_wrapper& operator=(async_rw_mutex_access_wrapper const&) = default;
    };

    /// \brief A wrapper for read-write access granted by a \p void \ref async_rw_mutex.
    ///
    /// The wrapper is move-only.
    template <typename Allocator>
    class async_rw_mutex_access_wrapper<void, void, async_rw_mutex_access_type::readwrite,
        Allocator>
    {
    private:
        using shared_state_ptr_type =
            pika::intrusive_ptr<detail::async_rw_mutex_shared_state<void, Allocator>>;
        shared_state_ptr_type state;

    public:
        async_rw_mutex_access_wrapper() = delete;
        explicit async_rw_mutex_access_wrapper(shared_state_ptr_type state)
          : state(std::move(state))
        {
        }
        async_rw_mutex_access_wrapper(async_rw_mutex_access_wrapper&&) = default;
        async_rw_mutex_access_wrapper& operator=(async_rw_mutex_access_wrapper&&) = default;
        async_rw_mutex_access_wrapper(async_rw_mutex_access_wrapper const&) = delete;
        async_rw_mutex_access_wrapper& operator=(async_rw_mutex_access_wrapper const&) = delete;
    };

    /// \brief Read-write mutex where access is granted to a value through senders.
    ///
    /// The wrapped value is accessed through \ref read and \ref readwrite, both of which return
    /// senders which send a wrapped value on the value channel when the wrapped value is safe to
    /// read or write.
    ///
    /// A read-write sender gives exclusive access to the wrapped value, while a read-only sender
    /// allows concurrent access to the value (with other read-only accesses).
    ///
    /// When the wrapped type is \p void, the mutex acts as a simple mutex around some externally
    /// managed resource. The mutex still allows read-write and read-only access when the type is \p
    /// void. The read-write wrapper types are move-only. The read-only wrapper types are copyable.
    ///
    /// The order in which senders signal a receiver is determined by the order in which the senders
    /// are retrieved from the mutex. Connecting and starting the senders is thread-safe.
    ///
    /// The mutex is move-only.
    ///
    /// \warning Because access to the wrapped value is granted in the order that it is requested
    /// from the mutex, there is a risk of deadlocks if senders of later accesses are started and
    /// waited for without starting senders of earlier accesses.
    ///
    /// \warning Retrieving senders from the mutex is not thread-safe. The senders of the mutex are
    /// intended to be accessed in synchronous code, while the access provided by the senders
    /// themselves are safe to access concurrently.
    ///
    /// \tparam ReadWriteT The type of the wrapped type.
    /// \tparam ReadT The type to use for read-only accesses of the wrapped type. Defaults to \ref
    /// ReadWriteT.
    /// \tparam Allocator The allocator to use for allocating the internal shared state.
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
        template <async_rw_mutex_access_type AccessType>
        struct sender;

    public:
        using read_type = void;
        using readwrite_type = void;

        using read_access_type = async_rw_mutex_access_wrapper<readwrite_type, read_type,
            async_rw_mutex_access_type::read, Allocator>;
        using readwrite_access_type = async_rw_mutex_access_wrapper<readwrite_type, read_type,
            async_rw_mutex_access_type::readwrite, Allocator>;

        using allocator_type = Allocator;

        explicit async_rw_mutex(allocator_type const& alloc = {})
          : alloc(alloc)
        {
        }
        async_rw_mutex(async_rw_mutex&&) = default;
        async_rw_mutex& operator=(async_rw_mutex&&) = default;
        async_rw_mutex(async_rw_mutex const&) = delete;
        async_rw_mutex& operator=(async_rw_mutex const&) = delete;

    private:
        using shared_state_type = detail::async_rw_mutex_shared_state<void, allocator_type>;

        // nvc++ is not able to see this typedef unless it's public
#if defined(PIKA_NVHPC_VERSION)
    public:
#endif
        using shared_state_ptr_type = pika::intrusive_ptr<shared_state_type>;

    private:
        shared_state_ptr_type allocate_shared()
        {
            using other_allocator = typename std::allocator_traits<
                allocator_type>::template rebind_alloc<shared_state_type>;
            using allocator_traits = std::allocator_traits<other_allocator>;
            using unique_ptr = std::unique_ptr<shared_state_type,
                pika::detail::allocator_deleter<other_allocator>>;

            other_allocator allocator(this->alloc);
            unique_ptr p(allocator_traits::allocate(allocator, 1),
                pika::detail::allocator_deleter<other_allocator>{allocator});
            new (p.get()) shared_state_type{this->alloc};
            return p.release();
        }

    public:
        sender<async_rw_mutex_access_type::read> read()
        {
            if (prev_access == async_rw_mutex_access_type::readwrite)
            {
                auto prev_state = std::move(state);
                state = allocate_shared();
                prev_access = async_rw_mutex_access_type::read;

                // Only the first access has no previous shared state. When
                // there is a previous state we set the next state so that the
                // value can be passed from the previous state to the next
                // state.
                if (PIKA_LIKELY(prev_state)) { prev_state->set_next_state(state); }
                else { state->done(nullptr); }
            }

            return {state};
        }

        sender<async_rw_mutex_access_type::readwrite> readwrite()
        {
            auto prev_state = std::move(state);
            state = allocate_shared();
            prev_access = async_rw_mutex_access_type::readwrite;

            // Only the first access has no previous shared state. When there is
            // a previous state we set the next state so that the value can be
            // passed from the previous state to the next state.
            if (PIKA_LIKELY(prev_state)) { prev_state->set_next_state(state); }
            else { state->done(nullptr); }

            return {state};
        }

    private:
        template <async_rw_mutex_access_type AccessType>
        struct sender
        {
            PIKA_STDEXEC_SENDER_CONCEPT

            shared_state_ptr_type state;

            using access_type = async_rw_mutex_access_wrapper<readwrite_type, read_type, AccessType,
                allocator_type>;
            template <template <typename...> class Tuple, template <typename...> class Variant>
            using value_types = Variant<Tuple<access_type>>;

            template <template <typename...> class Variant>
            using error_types = Variant<std::exception_ptr>;

            static constexpr bool sends_done = false;

            using completion_signatures = pika::execution::experimental::completion_signatures<
                pika::execution::experimental::set_value_t(access_type),
                pika::execution::experimental::set_error_t(std::exception_ptr)>;

            template <typename R>
            struct operation_state : detail::async_rw_mutex_operation_state_base
            {
                std::decay_t<R> r;
                shared_state_ptr_type state;

                template <typename R_>
                operation_state(R_&& r, shared_state_ptr_type state)
                  : r(std::forward<R_>(r))
                  , state(std::move(state))
                {
                }

                operation_state(operation_state&&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                void continuation() override
                {
                    try
                    {
                        pika::execution::experimental::set_value(
                            std::move(r), access_type{std::move(state)});
                    }
                    catch (...)
                    {
                        state.reset();
                        pika::execution::experimental::set_error(
                            std::move(r), std::current_exception());
                    }
                }

                void start() & noexcept
                {
                    PIKA_ASSERT_MSG(state,
                        "async_rw_lock::sender::operation_state state is empty, was the sender "
                        "already started?");

                    if (!state->add_op_state(this))
                    {
                        // There is no previous state on the first access or the
                        // previous state has already been released. We can run
                        // the continuation immediately.
                        continuation();
                    }
                }
            };

            template <typename R>
            auto connect(R&& r) &&
            {
                return operation_state<R>{std::forward<R>(r), std::move(state)};
            }
        };

        PIKA_NO_UNIQUE_ADDRESS allocator_type alloc;

        async_rw_mutex_access_type prev_access = async_rw_mutex_access_type::readwrite;

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
        /// \brief The type of read-only types accessed through the mutex.
        using read_type = std::decay_t<ReadT> const;

        /// \brief The type of read-write types accessed through the mutex.
        using readwrite_type = std::decay_t<ReadWriteT>;

        /// \brief The wrapper type sent by read-only-access senders.
        using read_access_type = async_rw_mutex_access_wrapper<readwrite_type, read_type,
            async_rw_mutex_access_type::read, Allocator>;

        /// \brief The wrapper type sent by read-write-access senders.
        using readwrite_access_type = async_rw_mutex_access_wrapper<readwrite_type, read_type,
            async_rw_mutex_access_type::readwrite, Allocator>;

        using allocator_type = Allocator;

        async_rw_mutex() = delete;

        /// \brief Construct a new mutex with the wrapped value initialized to \p u.
        template <typename U,
            typename = std::enable_if_t<!std::is_same<std::decay_t<U>, async_rw_mutex>::value>>
        explicit async_rw_mutex(U&& u, allocator_type const& alloc = {})
          : value(std::allocate_shared<readwrite_type, allocator_type>(alloc, std::forward<U>(u)))
          , alloc(alloc)
        {
        }
        async_rw_mutex(async_rw_mutex&&) = default;
        async_rw_mutex& operator=(async_rw_mutex&&) = default;
        async_rw_mutex(async_rw_mutex const&) = delete;
        async_rw_mutex& operator=(async_rw_mutex const&) = delete;
        /// \brief Destroy the mutex.
        ///
        /// The destructor does not wait or require that all accesses through senders have
        /// completed. The wrapped value is kept alive in a shared state managed by the senders,
        /// until the last access completes, or the destructor of the \ref async_rw_mutex runs,
        /// whichever happens later.
        ~async_rw_mutex() = default;

    private:
        using shared_state_type =
            detail::async_rw_mutex_shared_state<readwrite_type, allocator_type>;
        using value_ptr_type = std::shared_ptr<readwrite_type>;

        // nvc++ is not able to see this typedef unless it's public
#if defined(PIKA_NVHPC_VERSION)
    public:
#endif
        using shared_state_ptr_type = pika::intrusive_ptr<shared_state_type>;

    private:
        shared_state_ptr_type allocate_shared()
        {
            using other_allocator = typename std::allocator_traits<
                allocator_type>::template rebind_alloc<shared_state_type>;
            using allocator_traits = std::allocator_traits<other_allocator>;
            using unique_ptr = std::unique_ptr<shared_state_type,
                pika::detail::allocator_deleter<other_allocator>>;

            other_allocator allocator(this->alloc);
            unique_ptr p(allocator_traits::allocate(allocator, 1),
                pika::detail::allocator_deleter<other_allocator>{allocator});
            new (p.get()) shared_state_type{this->alloc, value};
            return p.release();
        }

    public:
        /// \brief Access the wrapped value in read-only mode through a sender.
        sender<async_rw_mutex_access_type::read> read()
        {
            if (prev_access == async_rw_mutex_access_type::readwrite)
            {
                auto prev_state = std::move(state);
                state = allocate_shared();
                prev_access = async_rw_mutex_access_type::read;

                // Only the first access has no previous shared state.
                if (PIKA_LIKELY(prev_state)) { prev_state->set_next_state(state); }
                else { state->done(nullptr); }
            }

            return {state};
        }

        /// \brief Access the wrapped value in read-write mode through a sender.
        sender<async_rw_mutex_access_type::readwrite> readwrite()
        {
            auto prev_state = std::move(state);
            state = allocate_shared();
            prev_access = async_rw_mutex_access_type::readwrite;

            // Only the first access has no previous shared state.
            if (PIKA_LIKELY(prev_state)) { prev_state->set_next_state(state); }
            else { state->done(nullptr); }

            return {state};
        }

    private:
        template <async_rw_mutex_access_type AccessType>
        struct sender
        {
            PIKA_STDEXEC_SENDER_CONCEPT

            shared_state_ptr_type state;

            using access_type = async_rw_mutex_access_wrapper<readwrite_type, read_type, AccessType,
                allocator_type>;
            template <template <typename...> class Tuple, template <typename...> class Variant>
            using value_types = Variant<Tuple<access_type>>;

            template <template <typename...> class Variant>
            using error_types = Variant<std::exception_ptr>;

            static constexpr bool sends_done = false;

            using completion_signatures = pika::execution::experimental::completion_signatures<
                pika::execution::experimental::set_value_t(access_type),
                pika::execution::experimental::set_error_t(std::exception_ptr)>;

            template <typename R>
            struct operation_state : detail::async_rw_mutex_operation_state_base
            {
                std::decay_t<R> r;
                shared_state_ptr_type state;

                template <typename R_>
                operation_state(R_&& r, shared_state_ptr_type state)
                  : r(std::forward<R_>(r))
                  , state(std::move(state))
                {
                }

                operation_state(operation_state&&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                void continuation() override
                {
                    try
                    {
                        pika::execution::experimental::set_value(
                            std::move(r), access_type{std::move(state)});
                    }
                    catch (...)
                    {
                        state.reset();
                        pika::execution::experimental::set_error(
                            std::move(r), std::current_exception());
                    }
                }

                void start() & noexcept
                {
                    PIKA_ASSERT_MSG(state,
                        "async_rw_lock::sender::operation_state state is empty, was the sender "
                        "already started?");

                    if (!state->add_op_state(this))
                    {
                        // There is no previous state on the first access or the
                        // previous state has already been released. We can run
                        // the continuation immediately.
                        continuation();
                    }
                }
            };

            template <typename R>
            auto connect(R&& r) &&
            {
                return operation_state<R>{std::forward<R>(r), std::move(state)};
            }

            template <typename R>
            auto connect(R&& r) const&
            {
                if constexpr (AccessType == async_rw_mutex_access_type::readwrite)
                {
                    static_assert(sizeof(R) == 0,
                        "senders returned from async_rw_mutex::readwrite are not l-lvalue "
                        "connectable");
                }

                return operation_state<R>{std::forward<R>(r), state};
            }
        };

        value_ptr_type value;
        PIKA_NO_UNIQUE_ADDRESS allocator_type alloc;

        async_rw_mutex_access_type prev_access = async_rw_mutex_access_type::readwrite;

        shared_state_ptr_type state;
    };
}    // namespace pika::execution::experimental
