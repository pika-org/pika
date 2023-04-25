//  Copyright (c) 2007-2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/concurrency/cache_line_data.hpp>
#include <pika/functional/function.hpp>
#include <pika/modules/errors.hpp>
#include <pika/threading_base/scheduler_mode.hpp>
#include <pika/threading_base/scheduler_state.hpp>
#include <pika/threading_base/thread_data.hpp>
#include <pika/threading_base/thread_init_data.hpp>
#include <pika/threading_base/thread_pool_base.hpp>
#include <pika/threading_base/thread_queue_init_parameters.hpp>
#include <pika/threading_base/threading_base_fwd.hpp>
#if defined(PIKA_HAVE_SCHEDULER_LOCAL_STORAGE)
# include <pika/coroutines/detail/tss.hpp>
#endif

#include <fmt/format.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <pika/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace pika::threads::detail {
    enum class polling_status
    {
        /// Signals that a polling function currently has no more work to do
        idle = 0,
        /// Signals that a polling function still has outstanding work to
        /// poll for
        busy = 1
    };

    ///////////////////////////////////////////////////////////////////////////
    /// The scheduler_base defines the interface to be implemented by all
    /// scheduler policies
    struct PIKA_EXPORT scheduler_base
    {
    public:
        PIKA_NON_COPYABLE(scheduler_base);

    public:
        using pu_mutex_type = std::mutex;

        scheduler_base(std::size_t num_threads, char const* description = "",
            thread_queue_init_parameters thread_queue_init = {},
            scheduler_mode mode = scheduler_mode::nothing_special);

        virtual ~scheduler_base() = default;

        threads::detail::thread_pool_base* get_parent_pool() const
        {
            PIKA_ASSERT(parent_pool_ != nullptr);
            return parent_pool_;
        }

        void set_parent_pool(threads::detail::thread_pool_base* p)
        {
            PIKA_ASSERT(parent_pool_ == nullptr);
            parent_pool_ = p;
        }

        inline std::size_t global_to_local_thread_index(std::size_t n)
        {
            return n - parent_pool_->get_thread_offset();
        }

        inline std::size_t local_to_global_thread_index(std::size_t n)
        {
            return n + parent_pool_->get_thread_offset();
        }

        char const* get_description() const
        {
            return description_;
        }

        void idle_callback(std::size_t num_thread);

        /// This function gets called by the thread-manager whenever new work
        /// has been added, allowing the scheduler to reactivate one or more of
        /// possibly idling OS threads
        void do_some_work(std::size_t);

        virtual void suspend(std::size_t num_thread);
        virtual void resume(std::size_t num_thread);

        std::size_t select_active_pu(std::unique_lock<pu_mutex_type>& l, std::size_t num_thread,
            bool allow_fallback = false);

        // allow to access/manipulate states
        std::atomic<pika::runtime_state>& get_state(std::size_t num_thread);
        std::atomic<pika::runtime_state> const& get_state(std::size_t num_thread) const;
        void set_all_states(pika::runtime_state s);
        void set_all_states_at_least(pika::runtime_state s);

        // return whether all states are at least at the given one
        bool has_reached_state(pika::runtime_state s) const;
        bool is_state(pika::runtime_state s) const;
        std::pair<pika::runtime_state, pika::runtime_state> get_minmax_state() const;

        ///////////////////////////////////////////////////////////////////////
        // get/set scheduler mode
        scheduler_mode get_scheduler_mode() const
        {
            return mode_.data_.load(std::memory_order_relaxed);
        }

        // get/set scheduler mode
        bool has_scheduler_mode(scheduler_mode mode) const
        {
            return (mode_.data_.load(std::memory_order_relaxed) & mode) != scheduler_mode{};
        }

        // set mode flags that control scheduler behaviour
        // This set function is virtual so that flags may be overridden
        // by schedulers that do not support certain operations/modes.
        // All other mode set functions should call this one to ensure
        // that flags are always consistent
        virtual void set_scheduler_mode(scheduler_mode mode);

        // add a flag to the scheduler mode flags
        void add_scheduler_mode(scheduler_mode mode);

        // remove flag from scheduler mode
        void remove_scheduler_mode(scheduler_mode mode);

        // conditionally add or remove depending on set true/false
        void update_scheduler_mode(scheduler_mode mode, bool set);

        pu_mutex_type& get_pu_mutex(std::size_t num_thread)
        {
            PIKA_ASSERT(num_thread < pu_mtxs_.size());
            return pu_mtxs_[num_thread];
        }

        ///////////////////////////////////////////////////////////////////////
        // domain management
        std::size_t domain_from_local_thread_index(std::size_t n);

        // assumes queues use index 0..N-1 and correspond to the pool cores
        std::size_t num_domains(const std::size_t workers);

        // either threads in same domain, or not in same domain
        // depending on the predicate
        std::vector<std::size_t> domain_threads(std::size_t local_id,
            const std::vector<std::size_t>& ts,
            util::detail::function<bool(std::size_t, std::size_t)> pred);

#ifdef PIKA_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
        virtual std::uint64_t get_creation_time(bool reset) = 0;
        virtual std::uint64_t get_cleanup_time(bool reset) = 0;
#endif

#ifdef PIKA_HAVE_THREAD_STEALING_COUNTS
        virtual std::int64_t get_num_pending_misses(std::size_t num_thread, bool reset) = 0;
        virtual std::int64_t get_num_pending_accesses(std::size_t num_thread, bool reset) = 0;

        virtual std::int64_t get_num_stolen_from_pending(std::size_t num_thread, bool reset) = 0;
        virtual std::int64_t get_num_stolen_to_pending(std::size_t num_thread, bool reset) = 0;
        virtual std::int64_t get_num_stolen_from_staged(std::size_t num_thread, bool reset) = 0;
        virtual std::int64_t get_num_stolen_to_staged(std::size_t num_thread, bool reset) = 0;
#endif

        virtual std::int64_t get_queue_length(std::size_t num_thread = std::size_t(-1)) const = 0;

        virtual std::int64_t get_thread_count(threads::detail::thread_schedule_state state =
                                                  threads::detail::thread_schedule_state::unknown,
            execution::thread_priority priority = execution::thread_priority::default_,
            std::size_t num_thread = std::size_t(-1), bool reset = false) const = 0;

        // Queries whether a given core is idle
        virtual bool is_core_idle(std::size_t num_thread) const = 0;

        // Enumerate all matching threads
        virtual bool enumerate_threads(
            util::detail::function<bool(threads::detail::thread_id_type)> const& f,
            threads::detail::thread_schedule_state state =
                threads::detail::thread_schedule_state::unknown) const = 0;

        virtual void abort_all_suspended_threads() = 0;

        virtual bool cleanup_terminated(bool delete_all) = 0;
        virtual bool cleanup_terminated(std::size_t num_thread, bool delete_all) = 0;

        virtual void create_thread(threads::detail::thread_init_data& data,
            threads::detail::thread_id_ref_type* id, error_code& ec) = 0;

        virtual bool get_next_thread(std::size_t num_thread, bool running,
            threads::detail::thread_id_ref_type& thrd, bool enable_stealing) = 0;

        virtual void schedule_thread(threads::detail::thread_id_ref_type thrd,
            execution::thread_schedule_hint schedulehint, bool allow_fallback = false,
            execution::thread_priority priority = execution::thread_priority::normal) = 0;

        virtual void schedule_thread_last(threads::detail::thread_id_ref_type thrd,
            execution::thread_schedule_hint schedulehint, bool allow_fallback = false,
            execution::thread_priority priority = execution::thread_priority::normal) = 0;

        virtual void destroy_thread(threads::detail::thread_data* thrd) = 0;

        virtual bool wait_or_add_new(std::size_t num_thread, bool running,
            std::int64_t& idle_loop_count, bool enable_stealing, std::size_t& added) = 0;

        virtual void on_start_thread(std::size_t num_thread) = 0;
        virtual void on_stop_thread(std::size_t num_thread) = 0;
        virtual void on_error(std::size_t num_thread, std::exception_ptr const& e) = 0;

#ifdef PIKA_HAVE_THREAD_QUEUE_WAITTIME
        virtual std::int64_t get_average_thread_wait_time(
            std::size_t num_thread = std::size_t(-1)) const = 0;
        virtual std::int64_t get_average_task_wait_time(
            std::size_t num_thread = std::size_t(-1)) const = 0;
#endif

        virtual void reset_thread_distribution() {}

        std::ptrdiff_t get_stack_size(execution::thread_stacksize stacksize) const
        {
            if (stacksize == execution::thread_stacksize::current)
            {
                stacksize = threads::detail::get_self_stacksize_enum();
            }

            PIKA_ASSERT(stacksize != execution::thread_stacksize::current);

            switch (stacksize)
            {
            case execution::thread_stacksize::small_:
                return thread_queue_init_.small_stacksize_;

            case execution::thread_stacksize::medium:
                return thread_queue_init_.medium_stacksize_;

            case execution::thread_stacksize::large:
                return thread_queue_init_.large_stacksize_;

            case execution::thread_stacksize::huge:
                return thread_queue_init_.huge_stacksize_;

            case execution::thread_stacksize::nostack:
                return (std::numeric_limits<std::ptrdiff_t>::max)();

            default:
                PIKA_ASSERT_MSG(false, fmt::format("Invalid stack size {}", stacksize));
                break;
            }

            return thread_queue_init_.small_stacksize_;
        }

        using polling_function_ptr = polling_status (*)();
        using polling_work_count_function_ptr = std::size_t (*)();

        static polling_status null_polling_function()
        {
            return polling_status::idle;
        }

        static std::size_t null_polling_work_count_function()
        {
            return 0;
        }

        void set_mpi_polling_functions(
            polling_function_ptr mpi_func, polling_work_count_function_ptr mpi_work_count_func)
        {
            polling_function_mpi_.store(mpi_func, std::memory_order_relaxed);
            polling_work_count_function_mpi_.store(mpi_work_count_func, std::memory_order_relaxed);
        }

        void clear_mpi_polling_function()
        {
            polling_function_mpi_.store(&null_polling_function, std::memory_order_relaxed);
            polling_work_count_function_mpi_.store(
                &null_polling_work_count_function, std::memory_order_relaxed);
        }

        void set_cuda_polling_functions(
            polling_function_ptr cuda_func, polling_work_count_function_ptr cuda_work_count_func)
        {
            polling_function_cuda_.store(cuda_func, std::memory_order_relaxed);
            polling_work_count_function_cuda_.store(
                cuda_work_count_func, std::memory_order_relaxed);
        }

        void clear_cuda_polling_function()
        {
            polling_function_cuda_.store(&null_polling_function, std::memory_order_relaxed);
            polling_work_count_function_cuda_.store(
                &null_polling_work_count_function, std::memory_order_relaxed);
        }

        polling_status custom_polling_function() const
        {
            polling_status status = polling_status::idle;
#if defined(PIKA_HAVE_MODULE_ASYNC_MPI)
            if ((*polling_function_mpi_.load(std::memory_order_relaxed))() == polling_status::busy)
            {
                status = polling_status::busy;
            }
#endif
#if defined(PIKA_HAVE_MODULE_ASYNC_CUDA)
            if ((*polling_function_cuda_.load(std::memory_order_relaxed))() == polling_status::busy)
            {
                status = polling_status::busy;
            }
#endif
            return status;
        }

        std::size_t get_polling_work_count() const
        {
            std::size_t work_count = 0;
#if defined(PIKA_HAVE_MODULE_ASYNC_MPI)
            work_count += polling_work_count_function_mpi_.load(std::memory_order_relaxed)();
#endif
#if defined(PIKA_HAVE_MODULE_ASYNC_CUDA)
            work_count += polling_work_count_function_cuda_.load(std::memory_order_relaxed)();
#endif
            return work_count;
        }

    protected:
        // the scheduler mode, protected from false sharing
        pika::concurrency::detail::cache_line_data<std::atomic<scheduler_mode>> mode_;

#if defined(PIKA_HAVE_THREAD_MANAGER_IDLE_BACKOFF)
        // support for suspension on idle queues
        pu_mutex_type mtx_;
        std::condition_variable cond_;
        struct idle_backoff_data
        {
            std::uint32_t wait_count_;
            double max_idle_backoff_time_;
        };
        std::vector<pika::concurrency::detail::cache_line_data<idle_backoff_data>> wait_counts_;
#endif

        // support for suspension of pus
        std::vector<pu_mutex_type> suspend_mtxs_;
        std::vector<std::condition_variable> suspend_conds_;

        std::vector<pu_mutex_type> pu_mtxs_;

        std::vector<std::atomic<pika::runtime_state>> states_;
        char const* description_;

        thread_queue_init_parameters thread_queue_init_;

        // the pool that owns this scheduler
        threads::detail::thread_pool_base* parent_pool_;

        std::atomic<polling_function_ptr> polling_function_mpi_;
        std::atomic<polling_function_ptr> polling_function_cuda_;
        std::atomic<polling_work_count_function_ptr> polling_work_count_function_mpi_;
        std::atomic<polling_work_count_function_ptr> polling_work_count_function_cuda_;

#if defined(PIKA_HAVE_SCHEDULER_LOCAL_STORAGE)
    public:
        // manage scheduler-local data
        coroutines::detail::tss_data_node* find_tss_data(void const* key);
        void add_new_tss_node(void const* key,
            std::shared_ptr<coroutines::detail::tss_cleanup_function> const& func, void* tss_data);
        void erase_tss_node(void const* key, bool cleanup_existing);
        void* get_tss_data(void const* key);
        void set_tss_data(void const* key,
            std::shared_ptr<coroutines::detail::tss_cleanup_function> const& func, void* tss_data,
            bool cleanup_existing);

    protected:
        std::shared_ptr<coroutines::detail::tss_storage> thread_data_;
#endif
    };
}    // namespace pika::threads::detail

template <>
struct fmt::formatter<pika::threads::detail::scheduler_base> : fmt::formatter<std::string>
{
    template <typename FormatContext>
    auto format(pika::threads::detail::scheduler_base const& scheduler, FormatContext& ctx)
    {
        return fmt::formatter<std::string>::format(
            fmt::format(
                "{}({})", scheduler.get_description(), static_cast<const void*>(&scheduler)),
            ctx);
    }
};

#include <pika/config/warnings_suffix.hpp>
