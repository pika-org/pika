//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/concurrency/spinlock_pool.hpp>
#include <pika/coroutines/coroutine.hpp>
#include <pika/coroutines/detail/combined_tagged_state.hpp>
#include <pika/coroutines/thread_id_type.hpp>
#include <pika/debugging/backtrace.hpp>
#include <pika/execution_base/this_thread.hpp>
#include <pika/functional/function.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/logging.hpp>
#include <pika/modules/memory.hpp>
#include <pika/thread_support/atomic_count.hpp>
#include <pika/threading_base/thread_description.hpp>
#include <pika/threading_base/thread_init_data.hpp>
#if defined(PIKA_HAVE_APEX)
# include <pika/threading_base/external_timer.hpp>
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <forward_list>
#include <memory>
#include <mutex>
#include <stack>
#include <string>
#include <utility>

#include <pika/config/warnings_prefix.hpp>

////////////////////////////////////////////////////////////////////////////////
namespace pika::threads::detail {
    using get_locality_id_type = std::uint32_t(pika::error_code&);
    PIKA_EXPORT void set_get_locality_id(get_locality_id_type* f);
    PIKA_EXPORT std::uint32_t get_locality_id(pika::error_code&);

    ////////////////////////////////////////////////////////////////////////////
    class PIKA_EXPORT thread_data;    // forward declaration only

    ////////////////////////////////////////////////////////////////////////////
    /// The function \a get_self_id_data returns the data of the pika thread id
    /// associated with the current thread (or nullptr if the current thread is
    /// not a pika thread).
    PIKA_EXPORT thread_data* get_self_id_data();

    ////////////////////////////////////////////////////////////////////////////
    /// A \a thread is the representation of a ParalleX thread. It's a first
    /// class object in ParalleX. In our implementation this is a user level
    /// thread running on top of one of the OS threads spawned by the \a
    /// thread-manager.
    ///
    /// A \a thread encapsulates:
    ///  - A thread status word (see the functions \a thread#get_state and
    ///    \a thread#set_state)
    ///  - A function to execute (the thread function)
    ///  - A frame (in this implementation this is a block of memory used as
    ///    the threads stack)
    ///  - A block of registers (not implemented yet)
    ///
    /// Generally, \a threads are not created or executed directly. All
    /// functionality related to the management of \a threads is
    /// implemented by the thread-manager.
    class thread_data : public detail::thread_data_reference_counting
    {
    public:
        thread_data(thread_data const&) = delete;
        thread_data(thread_data&&) = delete;
        thread_data& operator=(thread_data const&) = delete;
        thread_data& operator=(thread_data&&) = delete;

    public:
        using spinlock_pool = pika::concurrency::detail::spinlock_pool<thread_data>;

        /// The get_state function queries the state of this thread instance.
        ///
        /// \returns        This function returns the current state of this
        ///                 thread. It will return one of the values as defined
        ///                 by the \a thread_state enumeration.
        ///
        /// \note           This function will be seldom used directly. Most of
        ///                 the time the state of a thread will be retrieved
        ///                 by using the function \a thread_manager#get_state.
        thread_state get_state(std::memory_order order = std::memory_order_acquire) const noexcept
        {
            return current_state_.load(order);
        }

        /// The set_state function changes the state of this thread instance.
        ///
        /// \param newstate [in] The new state to be set for the thread.
        ///
        /// \note           This function will be seldom used directly. Most of
        ///                 the time the state of a thread will have to be
        ///                 changed using the thread_manager. Moreover,
        ///                 changing the thread state using this function does
        ///                 not change its scheduling status. It only sets the
        ///                 thread's status word. To change the thread's
        ///                 scheduling status \a thread_manager#set_state should
        ///                 be used.
        // NOLINTBEGIN(bugprone-easily-swappable-parameters)
        thread_state set_state(thread_schedule_state state,
            thread_restart_state state_ex = thread_restart_state::unknown,
            std::memory_order load_order = std::memory_order_acquire,
            std::memory_order exchange_order = std::memory_order_seq_cst) noexcept
        // NOLINTEND(bugprone-easily-swappable-parameters)
        {
            thread_state prev_state = current_state_.load(load_order);

            for (;;)
            {
                thread_state tmp = prev_state;

                // ABA prevention for state only (not for state_ex)
                std::int64_t tag = tmp.tag();
                if (state != tmp.state())
                    ++tag;

                if (state_ex == thread_restart_state::unknown)
                    state_ex = tmp.state_ex();

                if (PIKA_LIKELY(current_state_.compare_exchange_strong(
                        tmp, thread_state(state, state_ex, tag), exchange_order)))
                {
                    return prev_state;
                }

                prev_state = tmp;
            }
        }

        bool set_state_tagged(thread_schedule_state newstate, thread_state& prev_state,
            thread_state& new_tagged_state,
            std::memory_order exchange_order = std::memory_order_seq_cst) noexcept
        {
            new_tagged_state = thread_state(newstate, prev_state.state_ex(), prev_state.tag() + 1);

            thread_state tmp = prev_state;
            return current_state_.compare_exchange_strong(tmp, new_tagged_state, exchange_order);
        }

        /// The restore_state function changes the state of this thread
        /// instance depending on its current state. It will change the state
        /// atomically only if the current state is still the same as passed
        /// as the second parameter. Otherwise it won't touch the thread state
        /// of this instance.
        ///
        /// \param newstate [in] The new state to be set for the thread.
        /// \param oldstate [in] The old state of the thread which still has to
        ///                 be the current state.
        ///
        /// \note           This function will be seldom used directly. Most of
        ///                 the time the state of a thread will have to be
        ///                 changed using the thread_manager. Moreover,
        ///                 changing the thread state using this function does
        ///                 not change its scheduling status. It only sets the
        ///                 thread's status word. To change the thread's
        ///                 scheduling status \a thread_manager#set_state should
        ///                 be used.
        ///
        /// \returns This function returns \a true if the state has been
        ///          changed successfully
        // NOLINTBEGIN(bugprone-easily-swappable-parameters)
        bool restore_state(thread_state new_state, thread_state old_state,
            std::memory_order load_order = std::memory_order_relaxed,
            std::memory_order load_exchange = std::memory_order_seq_cst) noexcept
        // NOLINTEND(bugprone-easily-swappable-parameters)
        {
            // ignore the state_ex while compare-exchanging
            thread_state current_state = current_state_.load(load_order);
            thread_restart_state state_ex = current_state.state_ex();

            // ABA prevention for state only (not for state_ex)
            std::int64_t tag = current_state.tag();
            if (new_state.state() != old_state.state())
                ++tag;

            thread_state old_tmp(old_state.state(), state_ex, old_state.tag());
            thread_state new_tmp(new_state.state(), state_ex, tag);

            return current_state_.compare_exchange_strong(old_tmp, new_tmp, load_exchange);
        }

        bool restore_state(thread_schedule_state new_state, thread_restart_state state_ex,
            thread_state old_state,
            std::memory_order load_exchange = std::memory_order_seq_cst) noexcept
        {
            // ABA prevention for state only (not for state_ex)
            std::int64_t tag = old_state.tag();
            if (new_state != old_state.state())
                ++tag;

            return current_state_.compare_exchange_strong(
                old_state, thread_state(new_state, state_ex, tag), load_exchange);
        }

    protected:
        /// The set_state function changes the extended state of this
        /// thread instance.
        ///
        /// \param newstate [in] The new extended state to be set for the
        ///                 thread.
        ///
        /// \note           This function will be seldom used directly. Most of
        ///                 the time the state of a thread will have to be
        ///                 changed using the thread_manager.
        thread_restart_state set_state_ex(thread_restart_state new_state) noexcept
        {
            thread_state prev_state = current_state_.load(std::memory_order_acquire);

            for (;;)
            {
                thread_state tmp = prev_state;

                if (PIKA_LIKELY(current_state_.compare_exchange_strong(
                        tmp, thread_state(tmp.state(), new_state, tmp.tag()))))
                {
                    return prev_state.state_ex();
                }

                prev_state = tmp;
            }
        }

    public:
        /// Return the id of the component this thread is running in
        constexpr std::uint64_t    // same as naming::address_type
        get_component_id() const noexcept
        {
            return 0;
        }

#if !defined(PIKA_HAVE_THREAD_DESCRIPTION)
        ::pika::detail::thread_description get_description() const
        {
            return ::pika::detail::thread_description("<unknown>");
        }

        ::pika::detail::thread_description set_description(
            ::pika::detail::thread_description /*value*/)
        {
            return ::pika::detail::thread_description("<unknown>");
        }

        ::pika::detail::thread_description get_lco_description() const
        {
            return ::pika::detail::thread_description("<unknown>");
        }
        ::pika::detail::thread_description set_lco_description(
            ::pika::detail::thread_description /*value*/)
        {
            return ::pika::detail::thread_description("<unknown>");
        }
#else
        ::pika::detail::thread_description get_description() const
        {
            std::lock_guard<pika::detail::spinlock> l(spinlock_pool::spinlock_for(this));
            return description_;
        }
        ::pika::detail::thread_description set_description(::pika::detail::thread_description value)
        {
            std::lock_guard<pika::detail::spinlock> l(spinlock_pool::spinlock_for(this));
            std::swap(description_, value);
            return value;
        }

        ::pika::detail::thread_description get_lco_description() const
        {
            std::lock_guard<pika::detail::spinlock> l(spinlock_pool::spinlock_for(this));
            return lco_description_;
        }
        ::pika::detail::thread_description set_lco_description(
            ::pika::detail::thread_description value)
        {
            std::lock_guard<pika::detail::spinlock> l(spinlock_pool::spinlock_for(this));
            std::swap(lco_description_, value);
            return value;
        }
#endif

#if !defined(PIKA_HAVE_THREAD_PARENT_REFERENCE)
        /// Return the locality of the parent thread
        constexpr std::uint32_t get_parent_locality_id() const noexcept
        {
            // this is the same as naming::invalid_locality_id
            return ~static_cast<std::uint32_t>(0);
        }

        /// Return the thread id of the parent thread
        constexpr thread_id_type get_parent_thread_id() const noexcept
        {
            return invalid_thread_id;
        }

        /// Return the phase of the parent thread
        constexpr std::size_t get_parent_thread_phase() const noexcept
        {
            return 0;
        }
#else
        /// Return the locality of the parent thread
        std::uint32_t get_parent_locality_id() const noexcept
        {
            return parent_locality_id_;
        }

        /// Return the thread id of the parent thread
        thread_id_type get_parent_thread_id() const noexcept
        {
            return parent_thread_id_;
        }

        /// Return the phase of the parent thread
        std::size_t get_parent_thread_phase() const noexcept
        {
            return parent_thread_phase_;
        }
#endif

#ifdef PIKA_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
        void set_marked_state(thread_schedule_state mark) const noexcept
        {
            marked_state_ = mark;
        }
        thread_schedule_state get_marked_state() const noexcept
        {
            return marked_state_;
        }
#endif

#if !defined(PIKA_HAVE_THREAD_BACKTRACE_ON_SUSPENSION)

# ifdef PIKA_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
        constexpr char const* get_backtrace() const noexcept
        {
            return nullptr;
        }
        char const* set_backtrace(char const*) noexcept
        {
            return nullptr;
        }
# else
        constexpr debug::detail::backtrace const* get_backtrace() const noexcept
        {
            return nullptr;
        }
        debug::detail::backtrace const* set_backtrace(debug::detail::backtrace const*) noexcept
        {
            return nullptr;
        }
# endif

#else    // defined(PIKA_HAVE_THREAD_BACKTRACE_ON_SUSPENSION

# ifdef PIKA_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
        char const* get_backtrace() const noexcept
        {
            std::lock_guard<pika::detail::spinlock> l(spinlock_pool::spinlock_for(this));
            return backtrace_;
        }
        char const* set_backtrace(char const* value) noexcept
        {
            std::lock_guard<pika::detail::spinlock> l(spinlock_pool::spinlock_for(this));

            char const* bt = backtrace_;
            backtrace_ = value;
            return bt;
        }
# else
        debug::detail::backtrace const* get_backtrace() const noexcept
        {
            std::lock_guard<pika::detail::spinlock> l(spinlock_pool::spinlock_for(this));
            return backtrace_;
        }
        debug::detail::backtrace const* set_backtrace(
            debug::detail::backtrace const* value) noexcept
        {
            std::lock_guard<pika::detail::spinlock> l(spinlock_pool::spinlock_for(this));

            debug::detail::backtrace const* bt = backtrace_;
            backtrace_ = value;
            return bt;
        }
# endif

        // Generate full backtrace for captured stack
        std::string backtrace()
        {
            std::lock_guard<pika::detail::spinlock> l(spinlock_pool::spinlock_for(this));

            std::string bt;
            if (0 != backtrace_)
            {
# ifdef PIKA_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
                bt = *backtrace_;
# else
                bt = backtrace_->trace();
# endif
            }
            return bt;
        }
#endif

        constexpr execution::thread_priority get_priority() const noexcept
        {
            return priority_;
        }
        void set_priority(execution::thread_priority priority) noexcept
        {
            priority_ = priority;
        }

        // handle thread interruption
        bool interruption_requested() const noexcept
        {
            std::lock_guard<pika::detail::spinlock> l(spinlock_pool::spinlock_for(this));
            return requested_interrupt_;
        }

        bool interruption_enabled() const noexcept
        {
            std::lock_guard<pika::detail::spinlock> l(spinlock_pool::spinlock_for(this));
            return enabled_interrupt_;
        }

        bool set_interruption_enabled(bool enable) noexcept
        {
            std::lock_guard<pika::detail::spinlock> l(spinlock_pool::spinlock_for(this));
            std::swap(enabled_interrupt_, enable);
            return enable;
        }

        void interrupt(bool flag = true)
        {
            std::unique_lock<pika::detail::spinlock> l(spinlock_pool::spinlock_for(this));
            if (flag && !enabled_interrupt_)
            {
                l.unlock();
                PIKA_THROW_EXCEPTION(pika::error::thread_not_interruptable,
                    "thread_data::interrupt", "interrupts are disabled for this thread");
                return;
            }
            requested_interrupt_ = flag;
        }

        bool interruption_point(bool throw_on_interrupt = true);

        bool add_thread_exit_callback(util::detail::function<void()> const& f);
        void run_thread_exit_callbacks();
        void free_thread_exit_callbacks();

        PIKA_FORCEINLINE bool is_stackless() const noexcept
        {
            return is_stackless_;
        }

        void destroy_thread() override;

        scheduler_base* get_scheduler_base() const noexcept
        {
            return scheduler_base_;
        }

        std::size_t get_last_worker_thread_num() const noexcept
        {
            return last_worker_thread_num_;
        }

        void set_last_worker_thread_num(std::size_t last_worker_thread_num) noexcept
        {
            last_worker_thread_num_ = last_worker_thread_num;
        }

        std::ptrdiff_t get_stack_size() const noexcept
        {
            return stacksize_;
        }

        execution::thread_stacksize get_stack_size_enum() const noexcept
        {
            return stacksize_enum_;
        }

        template <typename ThreadQueue>
        ThreadQueue& get_queue() noexcept
        {
            return *static_cast<ThreadQueue*>(queue_);
        }

        /// \brief Execute the thread function
        ///
        /// \returns        This function returns the thread state the thread
        ///                 should be scheduled from this point on. The thread
        ///                 manager will use the returned value to set the
        ///                 thread's scheduling status.
        inline coroutine_type::result_type operator()(
            pika::execution::this_thread::detail::agent_storage* agent_storage);

        virtual thread_id_type get_thread_id() const
        {
            return thread_id_type{const_cast<thread_data*>(this)};
        }

#if !defined(PIKA_HAVE_THREAD_PHASE_INFORMATION)
        virtual std::size_t get_thread_phase() const noexcept
        {
            return 0;
        }
#else
        virtual std::size_t get_thread_phase() const noexcept = 0;
#endif
        virtual std::size_t get_thread_data() const = 0;
        virtual std::size_t set_thread_data(std::size_t data) = 0;

        virtual void init() = 0;
        virtual void rebind(thread_init_data& init_data) = 0;

#if defined(PIKA_HAVE_APEX)
        std::shared_ptr<pika::detail::external_timer::task_wrapper> get_timer_data() const noexcept
        {
            return timer_data_;
        }
        void set_timer_data(
            std::shared_ptr<pika::detail::external_timer::task_wrapper> data) noexcept
        {
            timer_data_ = data;
        }
#endif

        // Construct a new \a thread
        thread_data(thread_init_data& init_data, void* queue, std::ptrdiff_t stacksize,
            bool is_stackless = false, thread_id_addref addref = thread_id_addref::yes);

        virtual ~thread_data() override;
        virtual void destroy() = 0;

    protected:
        void rebind_base(thread_init_data& init_data);

    private:
        mutable std::atomic<thread_state> current_state_;

        ///////////////////////////////////////////////////////////////////////
        // Debugging/logging information
#ifdef PIKA_HAVE_THREAD_DESCRIPTION
        ::pika::detail::thread_description description_;
        ::pika::detail::thread_description lco_description_;
#endif

#ifdef PIKA_HAVE_THREAD_PARENT_REFERENCE
        std::uint32_t parent_locality_id_;
        thread_id_type parent_thread_id_;
        std::size_t parent_thread_phase_;
#endif

#ifdef PIKA_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
        mutable thread_schedule_state marked_state_;
#endif

#ifdef PIKA_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
# ifdef PIKA_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
        char const* backtrace_;
# else
        debug::detail::backtrace const* backtrace_;
# endif
#endif
        ///////////////////////////////////////////////////////////////////////
        execution::thread_priority priority_;

        bool requested_interrupt_;
        bool enabled_interrupt_;
        bool ran_exit_funcs_;
        bool const is_stackless_;

        // Singly linked list (heap-allocated)
        std::forward_list<util::detail::function<void()>> exit_funcs_;

        // reference to scheduler which created/manages this thread
        scheduler_base* scheduler_base_;
        std::size_t last_worker_thread_num_;

        std::ptrdiff_t stacksize_;
        execution::thread_stacksize stacksize_enum_;

        void* queue_;

    public:
#if defined(PIKA_HAVE_APEX)
        std::shared_ptr<pika::detail::external_timer::task_wrapper> timer_data_;
#endif
    };

    PIKA_FORCEINLINE thread_data* get_thread_id_data(thread_id_ref_type const& tid)
    {
        return static_cast<thread_data*>(tid.get().get());
    }

    PIKA_FORCEINLINE thread_data* get_thread_id_data(thread_id_type const& tid)
    {
        return static_cast<thread_data*>(tid.get());
    }

    namespace detail {
        PIKA_EXPORT void set_self_ptr(thread_self*);
    }

    ///////////////////////////////////////////////////////////////////////
    /// The function \a get_self returns a reference to the (OS thread
    /// specific) self reference to the current pika thread.
    PIKA_EXPORT thread_self& get_self();

    /// The function \a get_self_ptr returns a pointer to the (OS thread
    /// specific) self reference to the current pika thread.
    PIKA_EXPORT thread_self* get_self_ptr();

    /// The function \a get_ctx_ptr returns a pointer to the internal data
    /// associated with each coroutine.
    PIKA_EXPORT thread_self_impl_type* get_ctx_ptr();

    /// The function \a get_self_ptr_checked returns a pointer to the (OS
    /// thread specific) self reference to the current pika thread.
    PIKA_EXPORT thread_self* get_self_ptr_checked(error_code& ec = throws);

    /// The function \a get_self_id returns the pika thread id of the current
    /// thread (or zero if the current thread is not a pika thread).
    PIKA_EXPORT thread_id_type get_self_id();

    /// The function \a get_parent_id returns the pika thread id of the
    /// current thread's parent (or zero if the current thread is not a
    /// pika thread).
    ///
    /// \note This function will return a meaningful value only if the
    ///       code was compiled with PIKA_HAVE_THREAD_PARENT_REFERENCE
    ///       being defined.
    PIKA_EXPORT thread_id_type get_parent_id();

    /// The function \a get_parent_phase returns the pika phase of the
    /// current thread's parent (or zero if the current thread is not a
    /// pika thread).
    ///
    /// \note This function will return a meaningful value only if the
    ///       code was compiled with PIKA_HAVE_THREAD_PARENT_REFERENCE
    ///       being defined.
    PIKA_EXPORT std::size_t get_parent_phase();

    /// The function \a get_self_stacksize returns the stack size of the
    /// current thread (or zero if the current thread is not a pika thread).
    PIKA_EXPORT std::ptrdiff_t get_self_stacksize();

    /// The function \a get_self_stacksize_enum returns the stack size of the /
    //current thread (or thread_stacksize::default if the current thread is not
    //a pika thread).
    PIKA_EXPORT execution::thread_stacksize get_self_stacksize_enum();

    /// The function \a get_parent_locality_id returns the id of the locality of
    /// the current thread's parent (or zero if the current thread is not a
    /// pika thread).
    ///
    /// \note This function will return a meaningful value only if the
    ///       code was compiled with PIKA_HAVE_THREAD_PARENT_REFERENCE
    ///       being defined.
    PIKA_EXPORT std::uint32_t get_parent_locality_id();

    /// The function \a get_self_component_id returns the lva of the
    /// component the current thread is acting on
    ///
    /// \note This function will return a meaningful value only if the
    ///       code was compiled with PIKA_HAVE_THREAD_TARGET_ADDRESS
    ///       being defined.
    PIKA_EXPORT std::uint64_t get_self_component_id();
}    // namespace pika::threads::detail

#include <pika/config/warnings_suffix.hpp>

#include <pika/threading_base/thread_data_stackful.hpp>
#include <pika/threading_base/thread_data_stackless.hpp>

namespace pika::threads::detail {
    PIKA_FORCEINLINE coroutine_type::result_type thread_data::operator()(
        pika::execution::this_thread::detail::agent_storage* agent_storage)
    {
        if (is_stackless())
        {
            return static_cast<thread_data_stackless*>(this)->call();
        }
        return static_cast<thread_data_stackful*>(this)->call(agent_storage);
    }
}    // namespace pika::threads::detail
