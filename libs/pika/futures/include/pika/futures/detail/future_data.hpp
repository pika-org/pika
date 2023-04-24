//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/async_base/launch_policy.hpp>
#include <pika/concurrency/spinlock.hpp>
#include <pika/coroutines/detail/get_stack_pointer.hpp>
#include <pika/datastructures/detail/small_vector.hpp>
#include <pika/errors/try_catch_exception_ptr.hpp>
#include <pika/functional/function.hpp>
#include <pika/futures/future_fwd.hpp>
#include <pika/futures/traits/future_access.hpp>
#include <pika/memory/intrusive_ptr.hpp>
#include <pika/modules/errors.hpp>
#include <pika/synchronization/condition_variable.hpp>
#include <pika/thread_support/assert_owns_lock.hpp>
#include <pika/thread_support/atomic_count.hpp>
#include <pika/threading_base/annotated_function.hpp>
#include <pika/threading_base/thread_helpers.hpp>
#include <pika/type_support/unused.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>

#include <pika/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace pika {
    enum class future_status
    {
        ready,
        timeout,
        deferred,
        uninitialized
    };
}    // namespace pika

///////////////////////////////////////////////////////////////////////////////
namespace pika::lcos::detail {

    using run_on_completed_error_handler_type =
        util::detail::function<void(std::exception_ptr const& e)>;
    PIKA_EXPORT void set_run_on_completed_error_handler(run_on_completed_error_handler_type f);

    ///////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct future_data;

    ///////////////////////////////////////////////////////////////////////
    struct future_data_refcnt_base;

    void intrusive_ptr_add_ref(future_data_refcnt_base* p) noexcept;
    void intrusive_ptr_release(future_data_refcnt_base* p) noexcept;

    ///////////////////////////////////////////////////////////////////////
    struct PIKA_EXPORT future_data_refcnt_base
    {
        // future shared states are non-copyable and non-movable
        future_data_refcnt_base(future_data_refcnt_base const&) = delete;
        future_data_refcnt_base(future_data_refcnt_base&&) = delete;
        future_data_refcnt_base& operator=(future_data_refcnt_base const&) = delete;
        future_data_refcnt_base& operator=(future_data_refcnt_base&&) = delete;

    public:
        using completed_callback_type = util::detail::unique_function<void()>;
        using completed_callback_vector_type =
            pika::detail::small_vector<completed_callback_type, 1>;

        using has_future_data_refcnt_base = void;

        virtual ~future_data_refcnt_base();

        virtual void set_on_completed(completed_callback_type) = 0;

        virtual bool requires_delete() noexcept
        {
            return 0 == --count_;
        }

        virtual void destroy() noexcept
        {
            delete this;
        }

        // This is a tag type used to convey the information that the caller is
        // _not_ going to addref the future_data instance
        struct init_no_addref
        {
        };

    protected:
        future_data_refcnt_base() noexcept
          : count_(0)
        {
        }
        explicit future_data_refcnt_base(init_no_addref) noexcept
          : count_(1)
        {
        }

        // reference counting
        friend void intrusive_ptr_add_ref(future_data_refcnt_base* p) noexcept;
        friend void intrusive_ptr_release(future_data_refcnt_base* p) noexcept;

        ::pika::detail::atomic_count count_;
    };

    /// support functions for pika::intrusive_ptr
    inline void intrusive_ptr_add_ref(future_data_refcnt_base* p) noexcept
    {
        ++p->count_;
    }
    inline void intrusive_ptr_release(future_data_refcnt_base* p) noexcept
    {
        if (p->requires_delete())
        {
            p->destroy();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct future_data_result
    {
        using type = Result;

        template <typename U>
        PIKA_FORCEINLINE static constexpr U&& set(U&& u) noexcept
        {
            return PIKA_FORWARD(U, u);
        }
    };

    template <typename Result>
    struct future_data_result<Result&>
    {
        using type = Result*;

        PIKA_FORCEINLINE static constexpr Result* set(Result* u) noexcept
        {
            return u;
        }

        PIKA_FORCEINLINE static constexpr Result* set(Result& u) noexcept
        {
            return &u;
        }
    };

    template <>
    struct future_data_result<void>
    {
        using type = util::detail::unused_type;

        PIKA_FORCEINLINE static constexpr util::detail::unused_type set(
            util::detail::unused_type u) noexcept
        {
            return u;
        }
    };

    template <typename Result>
    using future_data_result_t = typename future_data_result<Result>::type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename R>
    struct future_data_storage
    {
        using value_type = future_data_result_t<R>;
        using error_type = std::exception_ptr;

        // determine the required alignment, define aligned storage of proper
        // size
        static constexpr std::size_t max_alignment =
            (std::alignment_of_v<value_type> > std::alignment_of_v<error_type>) ?
            std::alignment_of_v<value_type> :
            std::alignment_of_v<error_type>;

        static constexpr std::size_t max_size =
            (sizeof(value_type) > sizeof(error_type)) ? sizeof(value_type) : sizeof(error_type);

        using type = std::aligned_storage_t<max_size, max_alignment>;
    };

    template <typename Result>
    using future_data_storage_t = typename future_data_storage<Result>::type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct future_data_base;

    template <>
    struct PIKA_EXPORT future_data_base<traits::detail::future_data_void> : future_data_refcnt_base
    {
        using mutex_type = pika::detail::spinlock;

        future_data_base() noexcept
          : state_(empty)
        {
        }

        explicit future_data_base(init_no_addref no_addref) noexcept
          : future_data_refcnt_base(no_addref)
          , state_(empty)
        {
        }

        using future_data_refcnt_base::completed_callback_type;
        using future_data_refcnt_base::completed_callback_vector_type;
        using result_type = util::detail::unused_type;
        using init_no_addref = future_data_refcnt_base::init_no_addref;

        virtual ~future_data_base();

        enum state
        {
            empty = 0,
            ready = 1,
            value = 2 | ready,
            exception = 4 | ready
        };

        /// Return whether or not the data is available for this
        /// \a future.
        bool is_ready(std::memory_order order = std::memory_order_acquire) const noexcept
        {
            return (state_.load(order) & ready) != 0;
        }

        bool has_value() const noexcept
        {
            return state_.load(std::memory_order_acquire) == value;
        }

        bool has_exception() const noexcept
        {
#if defined(__GNUC__) && !defined(__clang__)
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
            return state_.load(std::memory_order_acquire) == exception;
#if defined(__GNUC__) && !defined(__clang__)
# pragma GCC diagnostic pop
#endif
        }

        virtual void execute_deferred(error_code& /*ec*/ = throws) {}

        // cancellation is disabled by default
        virtual bool cancelable() const noexcept
        {
            return false;
        }
        virtual void cancel()
        {
            PIKA_THROW_EXCEPTION(pika::error::future_does_not_support_cancellation,
                "future_data_base::cancel", "this future does not support cancellation");
        }

        result_type* get_result_void(void const* storage, error_code& ec = throws);
        virtual result_type* get_result_void(error_code& ec = throws) = 0;

        virtual void set_exception(std::exception_ptr data) = 0;

        // continuation support

        // deferred execution of a given continuation
        static void run_on_completed(completed_callback_type&& on_completed) noexcept;
        static void run_on_completed(completed_callback_vector_type&& on_completed) noexcept;

        // make sure continuation invocation does not recurse deeper than
        // allowed
        template <typename Callback>
        static void handle_on_completed(Callback&& on_completed);

        /// Set the callback which needs to be invoked when the future becomes
        /// ready. If the future is ready the function will be invoked
        /// immediately.
        void set_on_completed(completed_callback_type data_sink) override;

        virtual state wait(error_code& ec = throws);

        virtual pika::future_status wait_until(
            std::chrono::steady_clock::time_point const& abs_time, error_code& ec = throws);

        virtual std::exception_ptr get_exception_ptr() const = 0;

        virtual std::string const& get_registered_name() const
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status,
                "future_data_base::get_registered_name",
                "this future does not support name registration");
        }
        virtual void set_registered_name(std::string /*name*/)
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status,
                "future_data_base::set_registered_name",
                "this future does not support name registration");
        }
        virtual bool register_as(std::string /*name*/, bool /*manage_lifetime*/)
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status, "future_data_base::register_as",
                "this future does not support name registration");
        }

    protected:
        mutable mutex_type mtx_;
        std::atomic<state> state_;    // current state
        completed_callback_vector_type on_completed_;
        pika::detail::condition_variable cond_;    // threads waiting in read
    };

    struct in_place
    {
    };

    template <typename Result>
    struct future_data_base : future_data_base<traits::detail::future_data_void>
    {
    private:
        static void construct(void* p)
        {
            ::new (p) result_type();
        }

        template <typename T, typename... Ts>
        static void construct(void* p, T&& t, Ts&&... ts)
        {
            ::new (p) result_type(
                future_data_result<Result>::set(PIKA_FORWARD(T, t)), PIKA_FORWARD(Ts, ts)...);
        }

    public:
        using result_type = future_data_result_t<Result>;
        using base_type = future_data_base<traits::detail::future_data_void>;
        using init_no_addref = typename base_type::init_no_addref;
        using completed_callback_type = typename base_type::completed_callback_type;
        using completed_callback_vector_type = typename base_type::completed_callback_vector_type;

    protected:
        using mutex_type = typename base_type::mutex_type;

    public:
        future_data_base() = default;

        explicit future_data_base(init_no_addref no_addref) noexcept
          : base_type(no_addref)
        {
        }

        template <typename... Ts>
        future_data_base(init_no_addref no_addref, in_place, Ts&&... ts)
          : base_type(no_addref)
        {
            result_type* value_ptr = reinterpret_cast<result_type*>(&storage_);
            construct(value_ptr, PIKA_FORWARD(Ts, ts)...);
            state_.store(value, std::memory_order_relaxed);
        }

        future_data_base(init_no_addref no_addref, std::exception_ptr const& e)
          : base_type(no_addref)
        {
            std::exception_ptr* exception_ptr = reinterpret_cast<std::exception_ptr*>(&storage_);
            ::new ((void*) exception_ptr) std::exception_ptr(e);
            state_.store(exception, std::memory_order_relaxed);
        }
        future_data_base(init_no_addref no_addref, std::exception_ptr&& e)
          : base_type(no_addref)
        {
            std::exception_ptr* exception_ptr = reinterpret_cast<std::exception_ptr*>(&storage_);
            ::new ((void*) exception_ptr) std::exception_ptr(PIKA_MOVE(e));
            state_.store(exception, std::memory_order_relaxed);
        }

        ~future_data_base() noexcept override
        {
            reset();
        }

        /// Get the result of the requested action. This call blocks (yields
        /// control) if the result is not ready. As soon as the result has been
        /// returned and the waiting thread has been re-scheduled by the thread
        /// manager the function will return.
        ///
        /// \param ec     [in,out] this represents the error status on exit,
        ///               if this is pre-initialized to \a pika#throws
        ///               the function will throw on error instead. If the
        ///               operation blocks and is aborted because the object
        ///               went out of scope, the code \a pika#yield_aborted is
        ///               set or thrown.
        ///
        /// \note         If there has been an error reported (using the action
        ///               \a base_lco#set_exception), this function will throw an
        ///               exception encapsulating the reported error code and
        ///               error description if <code>&ec == &throws</code>.
        virtual result_type* get_result(error_code& ec = throws)
        {
            if (get_result_void(ec) != nullptr)
            {
                return reinterpret_cast<result_type*>(&storage_);
            }
            return nullptr;
        }

        util::detail::unused_type* get_result_void(error_code& ec = throws) override
        {
            return base_type::get_result_void(&storage_, ec);
        }

        // Set the result of the requested action.
        template <typename... Ts>
        void set_value(Ts&&... ts)
        {
            // Note: it is safe to access the data store as no other thread
            //       should access it concurrently. There shouldn't be any
            //       threads attempting to read the value as the state is still
            //       empty. Also, there can be only one thread (this thread)
            //       attempting to set the value by definition.

            // set the data
            result_type* value_ptr = reinterpret_cast<result_type*>(&storage_);
            construct(value_ptr, PIKA_FORWARD(Ts, ts)...);

            // At this point the lock needs to be acquired to safely access the
            // registered continuations
            std::unique_lock<mutex_type> l(mtx_);

            // handle all threads waiting for the future to become ready
            auto on_completed = PIKA_MOVE(on_completed_);
            on_completed_.clear();

            // The value has been set, changing the state to 'value' at this
            // point signals to all other threads that this future is ready.
            state expected = empty;
            if (!state_.compare_exchange_strong(expected, value, std::memory_order_release))
            {
                // this future should be 'empty' still (it can't be made ready
                // more than once).
                l.unlock();
                PIKA_THROW_EXCEPTION(pika::error::promise_already_satisfied,
                    "future_data_base::set_value", "data has already been set for this future");
                return;
            }

            // Note: we use notify_one repeatedly instead of notify_all as we
            //       know: a) that most of the time we have at most one thread
            //       waiting on the future (most futures are not shared), and
            //       b) our implementation of condition_variable::notify_one
            //       relinquishes the lock before resuming the waiting thread
            //       which avoids suspension of this thread when it tries to
            //       re-lock the mutex while exiting from condition_variable::wait
            while (cond_.notify_one(PIKA_MOVE(l), execution::thread_priority::boost))
            {
                l = std::unique_lock<mutex_type>(mtx_);
            }

            // Note: cv.notify_one() above 'consumes' the lock 'l' and leaves
            //       it unlocked when returning.

            // invoke the callback (continuation) function
            if (!on_completed.empty())
            {
                handle_on_completed(PIKA_MOVE(on_completed));
            }
        }

        void set_exception(std::exception_ptr data) override
        {
            // Note: it is safe to access the data store as no other thread
            //       should access it concurrently. There shouldn't be any
            //       threads attempting to read the value as the state is still
            //       empty. Also, there can be only one thread (this thread)
            //       attempting to set the value by definition.

            // set the data
            std::exception_ptr* exception_ptr = reinterpret_cast<std::exception_ptr*>(&storage_);
            ::new ((void*) exception_ptr) std::exception_ptr(PIKA_MOVE(data));

            // At this point the lock needs to be acquired to safely access the
            // registered continuations
            std::unique_lock<mutex_type> l(mtx_);

            // handle all threads waiting for the future to become ready
            auto on_completed = PIKA_MOVE(on_completed_);
            on_completed_.clear();

            // The value has been set, changing the state to 'exception' at this
            // point signals to all other threads that this future is ready.
            state expected = empty;
            if (!state_.compare_exchange_strong(expected, exception, std::memory_order_release))
            {
                // this future should be 'empty' still (it can't be made ready
                // more than once).
                l.unlock();
                PIKA_THROW_EXCEPTION(pika::error::promise_already_satisfied,
                    "future_data_base::set_exception", "data has already been set for this future");
                return;
            }

            // Note: we use notify_one repeatedly instead of notify_all as we
            //       know: a) that most of the time we have at most one thread
            //       waiting on the future (most futures are not shared), and
            //       b) our implementation of condition_variable::notify_one
            //       relinquishes the lock before resuming the waiting thread
            //       which avoids suspension of this thread when it tries to
            //       re-lock the mutex while exiting from condition_variable::wait
            while (cond_.notify_one(PIKA_MOVE(l), execution::thread_priority::boost))
            {
                l = std::unique_lock<mutex_type>(mtx_);
            }

            // Note: cv.notify_one() above 'consumes' the lock 'l' and leaves
            //       it unlocked when returning.

            // invoke the callback (continuation) function
            if (!on_completed.empty())
            {
                handle_on_completed(PIKA_MOVE(on_completed));
            }
        }

        // helper functions for setting data (if successful) or the error (if
        // non-successful)
        template <typename T>
        void set_data(T&& result)
        {
            pika::detail::try_catch_exception_ptr([&]() { set_value(PIKA_FORWARD(T, result)); },
                [&](std::exception_ptr ep) { set_exception(PIKA_MOVE(ep)); });
        }

        // trigger the future with the given error condition
        void set_error(error e, char const* f, char const* msg)
        {
            pika::detail::try_catch_exception_ptr([&]() { PIKA_THROW_EXCEPTION(e, f, "{}", msg); },
                [&](std::exception_ptr ep) { set_exception(PIKA_MOVE(ep)); });
        }

        /// Reset the promise to allow to restart an asynchronous
        /// operation. Allows any subsequent set_data operation to succeed.
        void reset(error_code& /*ec*/ = throws)
        {
            // no locking is required as semantics guarantee a single writer
            // and no reader

            // release any stored data and callback functions
            switch (state_.exchange(empty))
            {
            case value:
            {
                result_type* value_ptr = reinterpret_cast<result_type*>(&storage_);
                value_ptr->~result_type();
                break;
            }
            case exception:
            {
                std::exception_ptr* exception_ptr =
                    reinterpret_cast<std::exception_ptr*>(&storage_);
                exception_ptr->~exception_ptr();
                break;
            }
            default:
                break;
            }

            on_completed_.clear();
        }

        std::exception_ptr get_exception_ptr() const override
        {
            PIKA_ASSERT(state_.load(std::memory_order_acquire) == exception);
            return *reinterpret_cast<std::exception_ptr const*>(&storage_);
        }

    protected:
        using base_type::mtx_;
        using base_type::on_completed_;
        using base_type::state_;

    private:
        using base_type::cond_;
        future_data_storage_t<Result> storage_;
    };

    ///////////////////////////////////////////////////////////////////////////
    // Customization point to have the ability for creating distinct shared
    // states depending on the value type held.
    template <typename Result>
    struct future_data : future_data_base<Result>
    {
        using init_no_addref = typename future_data_base<Result>::init_no_addref;

        future_data() = default;

        explicit future_data(init_no_addref no_addref) noexcept
          : future_data_base<Result>(no_addref)
        {
        }

        template <typename... Ts>
        future_data(init_no_addref no_addref, in_place in_place, Ts&&... ts)
          : future_data_base<Result>(no_addref, in_place, PIKA_FORWARD(Ts, ts)...)
        {
        }

        future_data(init_no_addref no_addref, std::exception_ptr const& e)
          : future_data_base<Result>(no_addref, e)
        {
        }
        future_data(init_no_addref no_addref, std::exception_ptr&& e)
          : future_data_base<Result>(no_addref, PIKA_MOVE(e))
        {
        }

        ~future_data() noexcept override = default;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename Allocator, typename Derived = void>
    struct future_data_allocator : future_data<Result>
    {
        using init_no_addref = typename future_data<Result>::init_no_addref;

        using allocated_type =
            std::conditional_t<std::is_void_v<Derived>, future_data_allocator, Derived>;

        using other_allocator =
            typename std::allocator_traits<Allocator>::template rebind_alloc<allocated_type>;

        explicit future_data_allocator(other_allocator const& alloc) noexcept
          : future_data<Result>()
          , alloc_(alloc)
        {
        }

        future_data_allocator(init_no_addref no_addref, other_allocator const& alloc) noexcept
          : future_data<Result>(no_addref)
          , alloc_(alloc)
        {
        }

        template <typename... T>
        future_data_allocator(
            init_no_addref no_addref, in_place in_place, other_allocator const& alloc, T&&... ts)
          : future_data<Result>(no_addref, in_place, PIKA_FORWARD(T, ts)...)
          , alloc_(alloc)
        {
        }

        future_data_allocator(
            init_no_addref no_addref, std::exception_ptr const& e, other_allocator const& alloc)
          : future_data<Result>(no_addref, e)
          , alloc_(alloc)
        {
        }

        future_data_allocator(
            init_no_addref no_addref, std::exception_ptr&& e, other_allocator const& alloc)
          : future_data<Result>(no_addref, PIKA_MOVE(e))
          , alloc_(alloc)
        {
        }

    protected:
        void destroy() noexcept override
        {
            using traits = std::allocator_traits<other_allocator>;

            other_allocator alloc(alloc_);
            traits::destroy(alloc, static_cast<allocated_type*>(this));
            traits::deallocate(alloc, static_cast<allocated_type*>(this), 1);
        }

    private:
        other_allocator alloc_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct timed_future_data : future_data<Result>
    {
    public:
        using base_type = future_data<Result>;
        using result_type = typename base_type::result_type;

    public:
        timed_future_data() = default;

        template <typename Result_>
        timed_future_data(std::chrono::steady_clock::time_point const& abs_time, Result_&& init)
        {
            pika::intrusive_ptr<timed_future_data> this_(this);

            error_code ec;
            threads::detail::thread_init_data data(
                threads::detail::make_thread_function_nullary(
                    [this_ = PIKA_MOVE(this_), init = PIKA_FORWARD(Result_, init)]() {
                        this_->set_value(init);
                    }),
                "timed_future_data<Result>::timed_future_data", execution::thread_priority::boost,
                execution::thread_schedule_hint(), execution::thread_stacksize::current,
                threads::detail::thread_schedule_state::suspended, true);
            threads::detail::thread_id_ref_type id = threads::detail::register_thread(data, ec);
            if (ec)
            {
                // thread creation failed, report error to the new future
                this->base_type::set_exception(pika::detail::access_exception(ec));
                return;
            }

            // start new thread at given point in time
            threads::detail::set_thread_state(id.noref(), abs_time,
                threads::detail::thread_schedule_state::pending,
                threads::detail::thread_restart_state::timeout, execution::thread_priority::boost,
                true, ec);
            if (ec)
            {
                // thread scheduling failed, report error to the new future
                this->base_type::set_exception(pika::detail::access_exception(ec));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct task_base : future_data<Result>
    {
    protected:
        using base_type = future_data<Result>;
        using future_base_type = pika::intrusive_ptr<task_base>;
        using result_type = typename future_data<Result>::result_type;
        using init_no_addref = typename base_type::init_no_addref;

        using mutex_type = typename base_type::mutex_type;
        using base_type::mtx_;

    public:
        task_base() = default;

        explicit task_base(init_no_addref no_addref) noexcept
          : base_type(no_addref)
        {
        }

        void execute_deferred(error_code& /*ec*/ = throws) override
        {
            if (!started_test_and_set())
            {
                this->do_run();
            }
        }

        // retrieving the value
        result_type* get_result(error_code& ec = throws) override
        {
            if (!started_test_and_set())
            {
                this->do_run();
            }
            return this->future_data<Result>::get_result(ec);
        }

        // wait support
        typename base_type::state wait(error_code& ec = throws) override
        {
            if (!started_test_and_set())
            {
                this->do_run();
            }
            return this->future_data<Result>::wait(ec);
        }

        pika::future_status wait_until(
            std::chrono::steady_clock::time_point const& abs_time, error_code& ec = throws) override
        {
            if (!started_test())
            {
                return pika::future_status::deferred;    //-V110
            }
            return this->future_data<Result>::wait_until(abs_time, ec);
        }

    private:
        bool started_test() const noexcept
        {
            std::lock_guard<mutex_type> l(mtx_);
            return started_;
        }

        template <typename Lock>
        bool started_test_and_set_locked(Lock& l)
        {
            PIKA_ASSERT_OWNS_LOCK(l);
            if (started_)
            {
                return true;
            }
            started_ = true;
            return false;
        }

    protected:
        bool started_test_and_set()
        {
            std::lock_guard<mutex_type> l(mtx_);
            return started_test_and_set_locked(l);
        }

        void check_started()
        {
            std::unique_lock<mutex_type> l(mtx_);
            if (started_)
            {
                l.unlock();
                PIKA_THROW_EXCEPTION(pika::error::task_already_started, "task_base::check_started",
                    "this task has already been started");
                return;
            }
            started_ = true;
        }

    public:
        // run synchronously
        void run()
        {
            check_started();
            this->do_run();    // always on this thread
        }

        // run in a separate thread
        virtual threads::detail::thread_id_ref_type apply(
            threads::detail::thread_pool_base* /*pool*/, const char* /*annotation*/,
            launch /*policy*/, error_code& /*ec*/)
        {
            PIKA_ASSERT(false);    // shouldn't ever be called
            return threads::detail::invalid_thread_id;
        }

    protected:
        static void run_impl(future_base_type this_)
        {
            this_->do_run();
        }

    public:
        template <typename T>
        void set_data(T&& result)
        {
            this->future_data<Result>::set_data(PIKA_FORWARD(T, result));
        }

        void set_exception(std::exception_ptr e) override
        {
            this->future_data<Result>::set_exception(PIKA_MOVE(e));
        }

        virtual void do_run()
        {
            PIKA_ASSERT(false);    // shouldn't ever be called
        }

    protected:
        bool started_ = false;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct cancelable_task_base : task_base<Result>
    {
    protected:
        using base_type = task_base<Result>;
        using future_base_type = pika::intrusive_ptr<cancelable_task_base>;
        using result_type = typename future_data<Result>::result_type;
        using init_no_addref = typename task_base<Result>::init_no_addref;

        using mutex_type = typename base_type::mutex_type;
        using base_type::mtx_;

    protected:
        threads::detail::thread_id_type get_thread_id() const noexcept
        {
            std::lock_guard<mutex_type> l(mtx_);
            return id_;
        }
        void set_thread_id(threads::detail::thread_id_type id) noexcept
        {
            std::lock_guard<mutex_type> l(mtx_);
            id_ = id;
        }

    public:
        cancelable_task_base() noexcept
          : id_(threads::detail::invalid_thread_id)
        {
        }

        explicit cancelable_task_base(init_no_addref no_addref) noexcept
          : task_base<Result>(no_addref)
          , id_(threads::detail::invalid_thread_id)
        {
        }

    private:
        struct reset_id
        {
            reset_id(cancelable_task_base& target)
              : target_(target)
            {
                target.set_thread_id(threads::detail::get_self_id());
            }
            ~reset_id()
            {
                target_.set_thread_id(threads::detail::invalid_thread_id);
            }
            cancelable_task_base& target_;
        };

    protected:
        static void run_impl(future_base_type this_)
        {
            reset_id r(*this_);
            this_->do_run();
        }

    public:
        // cancellation support
        bool cancelable() const noexcept override
        {
            return true;
        }

        void cancel() override
        {
            std::unique_lock<mutex_type> l(mtx_);
            pika::detail::try_catch_exception_ptr(
                [&]() {
                    if (!this->started_)
                    {
                        PIKA_THROW_THREAD_INTERRUPTED_EXCEPTION();
                    }

                    if (this->is_ready())
                    {
                        return;    // nothing we can do
                    }

                    if (id_ != threads::detail::invalid_thread_id)
                    {
                        // interrupt the executing thread
                        threads::detail::interrupt_thread(id_);

                        this->started_ = true;

                        l.unlock();
                        this->set_error(pika::error::future_cancelled, "task_base<Result>::cancel",
                            "future has been canceled");
                    }
                    else
                    {
                        l.unlock();
                        PIKA_THROW_EXCEPTION(pika::error::future_can_not_be_cancelled,
                            "task_base<Result>::cancel", "future can't be canceled at this time");
                    }
                },
                [&](std::exception_ptr ep) {
                    this->started_ = true;
                    this->set_exception(ep);
                    std::rethrow_exception(PIKA_MOVE(ep));
                });
        }

    protected:
        threads::detail::thread_id_type id_;
    };
}    // namespace pika::lcos::detail

namespace pika::traits::detail {

    template <typename R, typename Allocator>
    struct shared_state_allocator<lcos::detail::future_data<R>, Allocator>
    {
        using type = lcos::detail::future_data_allocator<R, Allocator>;
    };
}    // namespace pika::traits::detail

#include <pika/config/warnings_suffix.hpp>
