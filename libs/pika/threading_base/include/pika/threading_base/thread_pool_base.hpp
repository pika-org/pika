//  Copyright (c)      2018 Mikael Simberg
//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/affinity/affinity_data.hpp>
#include <pika/concurrency/barrier.hpp>
#include <pika/functional/function.hpp>
#include <pika/modules/errors.hpp>
#include <pika/threading_base/callback_notifier.hpp>
#include <pika/threading_base/network_background_callback.hpp>
#include <pika/threading_base/scheduler_mode.hpp>
#include <pika/threading_base/scheduler_state.hpp>
#include <pika/threading_base/thread_init_data.hpp>
#include <pika/timing/steady_clock.hpp>
#include <pika/topology/cpu_mask.hpp>
#include <pika/topology/topology.hpp>

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <pika/config/warnings_prefix.hpp>

namespace pika::threads::detail {
    /// \brief Data structure which stores statistics collected by an
    ///        executor instance.
    struct executor_statistics
    {
        executor_statistics()
          : tasks_scheduled_(0)
          , tasks_completed_(0)
          , queue_length_(0)
        {
        }

        std::uint64_t tasks_scheduled_;
        std::uint64_t tasks_completed_;
        std::uint64_t queue_length_;
    };

    ///////////////////////////////////////////////////////////////////////
    enum executor_parameter
    {
        min_concurrency = 1,
        max_concurrency = 2,
        current_concurrency = 3
    };

    ///////////////////////////////////////////////////////////////////////
    // The interface below is used by the resource manager to
    // interact with the executor.
    struct PIKA_EXPORT manage_executor
    {
        virtual ~manage_executor() {}

        // Return the requested policy element
        virtual std::size_t get_policy_element(executor_parameter p, error_code& ec) const = 0;

        // Return statistics collected by this scheduler
        virtual void get_statistics(executor_statistics& stats, error_code& ec) const = 0;

        // Provide the given processing unit to the scheduler.
        virtual void add_processing_unit(
            std::size_t virt_core, std::size_t thread_num, error_code& ec) = 0;

        // Remove the given processing unit from the scheduler.
        virtual void remove_processing_unit(std::size_t thread_num, error_code& ec) = 0;

        // return the description string of the underlying scheduler
        virtual char const* get_description() const = 0;
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    struct pool_id_type
    {
        pool_id_type(std::size_t index, std::string const& name)
          : index_(index)
          , name_(name)
        {
        }

        std::size_t index() const
        {
            return index_;
        };
        std::string const& name() const
        {
            return name_;
        }

    private:
        std::size_t const index_;
        std::string const name_;
    };
    /// \endcond

    struct thread_pool_init_parameters
    {
        std::string const& name_;
        std::size_t index_;
        scheduler_mode mode_;
        std::size_t num_threads_;
        std::size_t thread_offset_;
        pika::threads::callback_notifier& notifier_;
        pika::detail::affinity_data const& affinity_data_;
        pika::threads::detail::network_background_callback_type const& network_background_callback_;
        std::size_t max_idle_loop_count_;
        std::size_t max_busy_loop_count_;
        std::size_t shutdown_check_count_;

        // NOLINTBEGIN(bugprone-easily-swappable-parameters)
        thread_pool_init_parameters(std::string const& name, std::size_t index, scheduler_mode mode,
            std::size_t num_threads, std::size_t thread_offset,
            pika::threads::callback_notifier& notifier,
            pika::detail::affinity_data const& affinity_data,
            pika::threads::detail::network_background_callback_type const&
                network_background_callback =
                    pika::threads::detail::network_background_callback_type(),
            std::size_t max_idle_loop_count = PIKA_IDLE_LOOP_COUNT_MAX,
            std::size_t max_busy_loop_count = PIKA_BUSY_LOOP_COUNT_MAX,
            std::size_t shutdown_check_count = 10)
          // NOLINTEND(bugprone-easily-swappable-parameters)
          : name_(name)
          , index_(index)
          , mode_(mode)
          , num_threads_(num_threads)
          , thread_offset_(thread_offset)
          , notifier_(notifier)
          , affinity_data_(affinity_data)
          , network_background_callback_(network_background_callback)
          , max_idle_loop_count_(max_idle_loop_count)
          , max_busy_loop_count_(max_busy_loop_count)
          , shutdown_check_count_(shutdown_check_count)
        {
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // note: this data structure has to be protected from races from the outside

    /// \brief The base class used to manage a pool of OS threads.
    class PIKA_EXPORT thread_pool_base
    {
    public:
        /// \cond NOINTERNAL
        thread_pool_base(thread_pool_init_parameters const& init);

        virtual ~thread_pool_base() = default;

        virtual void init(std::size_t num_threads, std::size_t threads_offset);

        virtual bool run(std::unique_lock<std::mutex>& l, std::size_t num_threads) = 0;

        virtual void stop(std::unique_lock<std::mutex>& l, bool blocking = true) = 0;

        virtual void wait() = 0;
        virtual bool is_busy() = 0;
        virtual bool is_idle() = 0;

        virtual void print_pool(std::ostream&) = 0;

        pool_id_type get_pool_id() const
        {
            return id_;
        }
        /// \endcond

        /// Suspends the given processing unit. Blocks until the processing unit
        /// has been suspended.
        ///
        /// \param virt_core [in] The processing unit on the the pool to be
        ///                  suspended. The processing units are indexed
        ///                  starting from 0.
        virtual void suspend_processing_unit_direct(
            std::size_t virt_core, error_code& ec = throws) = 0;

        /// Resumes the given processing unit. Blocks until the processing unit
        /// has been resumed.
        ///
        /// \param virt_core [in] The processing unit on the the pool to be resumed.
        ///                  The processing units are indexed starting from 0.
        virtual void resume_processing_unit_direct(
            std::size_t virt_core, error_code& ec = throws) = 0;

        /// Resumes the thread pool. Blocks until all OS threads on the thread pool
        /// have been resumed.
        ///
        /// \param ec [in,out] this represents the error status on exit, if this
        ///           is pre-initialized to \a pika#throws the function will
        ///           throw on error instead.
        virtual void resume_direct(error_code& ec = throws) = 0;

        /// Suspends the thread pool. Blocks until all OS threads on the thread pool
        /// have been suspended.
        ///
        /// \note A thread pool cannot be suspended from an pika thread running
        ///       on the pool itself.
        ///
        /// \param ec [in,out] this represents the error status on exit, if this
        ///           is pre-initialized to \a pika#throws the function will
        ///           throw on error instead.
        ///
        /// \throws pika::exception if called from an pika thread which is running
        ///         on the pool itself.
        virtual void suspend_direct(error_code& ec = throws) = 0;

    public:
        /// \cond NOINTERNAL
        virtual std::size_t get_os_thread_count() const = 0;

        virtual std::thread& get_os_thread_handle(std::size_t num_thread) = 0;

        virtual std::size_t get_active_os_thread_count() const;

        virtual void create_thread(
            thread_init_data& data, thread_id_ref_type& id, error_code& ec) = 0;
        virtual thread_id_ref_type create_work(thread_init_data& data, error_code& ec) = 0;

        virtual thread_state set_state(thread_id_type const& id, thread_schedule_state new_state,
            thread_restart_state new_state_ex, execution::thread_priority priority,
            error_code& ec) = 0;

        std::size_t get_pool_index() const
        {
            return id_.index();
        }
        std::string const& get_pool_name() const
        {
            return id_.name();
        }
        std::size_t get_thread_offset() const
        {
            return thread_offset_;
        }

        virtual scheduler_base* get_scheduler() const
        {
            return nullptr;
        }

        mask_type get_used_processing_units() const;
        hwloc_bitmap_ptr get_numa_domain_bitmap() const;

        // performance counters
#if defined(PIKA_HAVE_THREAD_CUMULATIVE_COUNTS)
        virtual std::int64_t get_executed_threads(std::size_t /*thread_num*/, bool /*reset*/)
        {
            return 0;
        }
        virtual std::int64_t get_executed_thread_phases(std::size_t /*thread_num*/, bool /*reset*/)
        {
            return 0;
        }
# if defined(PIKA_HAVE_THREAD_IDLE_RATES)
        virtual std::int64_t get_thread_phase_duration(std::size_t /*thread_num*/, bool /*reset*/)
        {
            return 0;
        }
        virtual std::int64_t get_thread_duration(std::size_t /*thread_num*/, bool /*reset*/)
        {
            return 0;
        }
        virtual std::int64_t get_thread_phase_overhead(std::size_t /*thread_num*/, bool /*reset*/)
        {
            return 0;
        }
        virtual std::int64_t get_thread_overhead(std::size_t /*thread_num*/, bool /*reset*/)
        {
            return 0;
        }
        virtual std::int64_t get_cumulative_thread_duration(
            std::size_t /*thread_num*/, bool /*reset*/)
        {
            return 0;
        }
        virtual std::int64_t get_cumulative_thread_overhead(
            std::size_t /*thread_num*/, bool /*reset*/)
        {
            return 0;
        }
# endif
#endif

        virtual std::int64_t get_cumulative_duration(std::size_t /*thread_num*/, bool /*reset*/)
        {
            return 0;
        }

#if defined(PIKA_HAVE_THREAD_IDLE_RATES)
        virtual std::int64_t avg_idle_rate_all(bool /*reset*/)
        {
            return 0;
        }
        virtual std::int64_t avg_idle_rate(std::size_t, bool)
        {
            return 0;
        }

# if defined(PIKA_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
        virtual std::int64_t avg_creation_idle_rate(std::size_t /*thread_num*/, bool /*reset*/)
        {
            return 0;
        }
        virtual std::int64_t avg_cleanup_idle_rate(std::size_t /*thread_num*/, bool /*reset*/)
        {
            return 0;
        }
# endif
#endif

        virtual std::int64_t get_queue_length(std::size_t, bool)
        {
            return 0;
        }

#if defined(PIKA_HAVE_THREAD_QUEUE_WAITTIME)
        virtual std::int64_t get_average_thread_wait_time(
            std::size_t /*thread_num*/, bool /*reset*/)
        {
            return 0;
        }
        virtual std::int64_t get_average_task_wait_time(std::size_t /*thread_num*/, bool /*reset*/)
        {
            return 0;
        }
#endif

#if defined(PIKA_HAVE_THREAD_STEALING_COUNTS)
        virtual std::int64_t get_num_pending_misses(std::size_t /*thread_num*/, bool /*reset*/)
        {
            return 0;
        }
        virtual std::int64_t get_num_pending_accesses(std::size_t /*thread_num*/, bool /*reset*/)
        {
            return 0;
        }

        virtual std::int64_t get_num_stolen_from_pending(std::size_t /*thread_num*/, bool /*reset*/)
        {
            return 0;
        }
        virtual std::int64_t get_num_stolen_to_pending(std::size_t /*thread_num*/, bool /*reset*/)
        {
            return 0;
        }
        virtual std::int64_t get_num_stolen_from_staged(std::size_t /*thread_num*/, bool /*reset*/)
        {
            return 0;
        }
        virtual std::int64_t get_num_stolen_to_staged(std::size_t /*thread_num*/, bool /*reset*/)
        {
            return 0;
        }
#endif

        virtual std::int64_t get_thread_count(thread_schedule_state /*state*/,
            execution::thread_priority /*priority*/, std::size_t /*num_thread*/, bool /*reset*/)
        {
            return 0;
        }

        virtual std::int64_t get_idle_core_count() const
        {
            return 0;
        }

        virtual void get_idle_core_mask(mask_type&) const {}

        std::int64_t get_thread_count_unknown(std::size_t num_thread, bool reset)
        {
            return get_thread_count(thread_schedule_state::unknown,
                execution::thread_priority::default_, num_thread, reset);
        }
        std::int64_t get_thread_count_active(std::size_t num_thread, bool reset)
        {
            return get_thread_count(thread_schedule_state::active,
                execution::thread_priority::default_, num_thread, reset);
        }
        std::int64_t get_thread_count_pending(std::size_t num_thread, bool reset)
        {
            return get_thread_count(thread_schedule_state::pending,
                execution::thread_priority::default_, num_thread, reset);
        }
        std::int64_t get_thread_count_suspended(std::size_t num_thread, bool reset)
        {
            return get_thread_count(thread_schedule_state::suspended,
                execution::thread_priority::default_, num_thread, reset);
        }
        std::int64_t get_thread_count_terminated(std::size_t num_thread, bool reset)
        {
            return get_thread_count(thread_schedule_state::terminated,
                execution::thread_priority::default_, num_thread, reset);
        }
        std::int64_t get_thread_count_staged(std::size_t num_thread, bool reset)
        {
            return get_thread_count(thread_schedule_state::staged,
                execution::thread_priority::default_, num_thread, reset);
        }

        virtual std::int64_t get_scheduler_utilization() const = 0;

        virtual std::int64_t get_idle_loop_count(std::size_t num, bool reset) = 0;
        virtual std::int64_t get_busy_loop_count(std::size_t num, bool reset) = 0;

        ///////////////////////////////////////////////////////////////////////
        virtual bool enumerate_threads(util::detail::function<bool(thread_id_type)> const& /*f*/,
            thread_schedule_state /*state*/ = thread_schedule_state::unknown) const
        {
            return false;
        }

        virtual void reset_thread_distribution() {}

        virtual void abort_all_suspended_threads() {}
        virtual bool cleanup_terminated(bool /*delete_all*/)
        {
            return false;
        }

        virtual pika::runtime_state get_state() const = 0;
        virtual pika::runtime_state get_state(std::size_t num_thread) const = 0;

        virtual bool has_reached_state(pika::runtime_state s) const = 0;

        virtual void do_some_work(std::size_t /*num_thread*/) {}

        virtual void report_error(std::size_t global_thread_num, std::exception_ptr const& e)
        {
            notifier_.on_error(global_thread_num, e);
        }

        double timestamp_scale() const
        {
            return timestamp_scale_;
        }
        /// \endcond

    protected:
        /// \cond NOINTERNAL
        void init_pool_time_scale();
        /// \endcond

    protected:
        /// \cond NOINTERNAL
        pool_id_type id_;

        // The thread_offset is equal to the accumulated number of
        // threads in all pools preceding this pool
        // in the thread indexation. That means, that in order to know
        // the global index of a thread it owns, the pool has to compute:
        // global index = thread_offset_ + local index.
        std::size_t thread_offset_;

        pika::detail::affinity_data const& affinity_data_;

        // scale timestamps to nanoseconds
        double timestamp_scale_;

        // callback functions to invoke at start, stop, and error
        threads::callback_notifier& notifier_;
        /// \endcond
    };
}    // namespace pika::threads::detail

template <>
struct fmt::formatter<pika::threads::detail::thread_pool_base> : fmt::formatter<std::string>
{
    template <typename FormatContext>
    auto format(pika::threads::detail::thread_pool_base const& thread_pool, FormatContext& ctx)
    {
        auto id = thread_pool.get_pool_id();
        return fmt::formatter<std::string>::format(
            fmt::format("{}({})", id.name(), id.index()), ctx);
    }
};

#include <pika/config/warnings_suffix.hpp>
