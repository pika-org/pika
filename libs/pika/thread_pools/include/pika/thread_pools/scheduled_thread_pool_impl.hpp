//  Copyright (c) 2017 Shoshana Jakobovits
//  Copyright (c) 2007-2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/affinity/affinity_data.hpp>
#include <pika/assert.hpp>
#include <pika/concurrency/barrier.hpp>
#include <pika/execution_base/this_thread.hpp>
#include <pika/functional/deferred_call.hpp>
#include <pika/functional/detail/invoke.hpp>
#include <pika/logging.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/schedulers.hpp>
#include <pika/thread_pools/scheduled_thread_pool.hpp>
#include <pika/thread_pools/scheduling_loop.hpp>
#include <pika/threading_base/callback_notifier.hpp>
#include <pika/threading_base/create_thread.hpp>
#include <pika/threading_base/create_work.hpp>
#include <pika/threading_base/scheduler_base.hpp>
#include <pika/threading_base/scheduler_mode.hpp>
#include <pika/threading_base/scheduler_state.hpp>
#include <pika/threading_base/set_thread_state.hpp>
#include <pika/threading_base/thread_data.hpp>
#include <pika/threading_base/thread_helpers.hpp>
#include <pika/threading_base/thread_num_tss.hpp>
#include <pika/topology/topology.hpp>

#include <algorithm>
#include <atomic>
#ifdef PIKA_HAVE_MAX_CPU_COUNT
# include <bitset>
#endif
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika::threads::detail {
    ///////////////////////////////////////////////////////////////////////////
    struct manage_active_thread_count
    {
        manage_active_thread_count(std::atomic<long>& counter)
          : counter_(counter)
        {
        }
        ~manage_active_thread_count() { --counter_; }

        std::atomic<long>& counter_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    struct init_tss_helper
    {
        // NOLINTBEGIN(bugprone-easily-swappable-parameters)
        init_tss_helper(scheduled_thread_pool<Scheduler>& pool, std::size_t local_thread_num,
            std::size_t global_thread_num)
          // NOLINTEND(bugprone-easily-swappable-parameters)
          : pool_(pool)
          , local_thread_num_(local_thread_num)
          , global_thread_num_(global_thread_num)
        {
            pool.notifier_.on_start_thread(
                local_thread_num_, global_thread_num_, pool_.get_pool_id().name().c_str(), "");
            pool.sched_->Scheduler::on_start_thread(local_thread_num_);
        }
        ~init_tss_helper()
        {
            pool_.sched_->Scheduler::on_stop_thread(local_thread_num_);
            pool_.notifier_.on_stop_thread(
                local_thread_num_, global_thread_num_, pool_.get_pool_id().name().c_str(), "");
        }

        scheduled_thread_pool<Scheduler>& pool_;
        std::size_t local_thread_num_;
        std::size_t global_thread_num_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    scheduled_thread_pool<Scheduler>::scheduled_thread_pool(
        std::unique_ptr<Scheduler> sched, thread_pool_init_parameters const& init)
      : thread_pool_base(init)
      , sched_(std::move(sched))
      , thread_count_(0)
      , max_idle_loop_count_(init.max_idle_loop_count_)
      , max_busy_loop_count_(init.max_busy_loop_count_)
      , shutdown_check_count_(init.shutdown_check_count_)
    {
        sched_->set_parent_pool(this);
    }

    template <typename Scheduler>
    scheduled_thread_pool<Scheduler>::~scheduled_thread_pool()
    {
        if (!threads_.empty())
        {
            if (!sched_->Scheduler::has_reached_state(runtime_state::suspended))
            {
                // still running
                std::mutex mtx;
                std::unique_lock<std::mutex> l(mtx);
                stop_locked(l);
            }
            threads_.clear();
        }
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::print_pool(std::ostream& os)
    {
        os << "[pool \"" << id_.name() << "\", #" << id_.index()    //-V128
           << "] with scheduler " << sched_->Scheduler::get_scheduler_name() << "\n"
           << "is running on PUs : \n";
        os << pika::threads::detail::to_string(get_used_processing_units())
#ifdef PIKA_HAVE_MAX_CPU_COUNT
           << " " << std::bitset<PIKA_HAVE_MAX_CPU_COUNT>(get_used_processing_units())
#endif
           << '\n';
        os << "on numa domains : \n" << get_numa_domain_bitmap() << '\n';
        os << "pool offset : \n" << std::dec << this->thread_offset_ << "\n";
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::report_error(
        std::size_t global_thread_num, std::exception_ptr const& e)
    {
        sched_->Scheduler::set_all_states_at_least(runtime_state::terminating);
        this->thread_pool_base::report_error(global_thread_num, e);
        sched_->Scheduler::on_error(global_thread_num, e);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    pika::runtime_state scheduled_thread_pool<Scheduler>::get_state() const
    {
        // This function might get called from within background_work inside the
        // os executors
        if (thread_count_ != 0)
        {
            std::size_t num_thread = get_local_thread_num_tss();

            // Local thread number may be valid, but the thread may not yet be
            // up.
            if (num_thread != std::size_t(-1) &&
                num_thread < static_cast<std::size_t>(thread_count_))
                return get_state(num_thread);
        }
        return sched_->Scheduler::get_minmax_state().second;
    }

    template <typename Scheduler>
    pika::runtime_state scheduled_thread_pool<Scheduler>::get_state(std::size_t num_thread) const
    {
        PIKA_ASSERT(num_thread != std::size_t(-1));
        return sched_->Scheduler::get_state(num_thread).load();
    }

    template <typename Scheduler>
    bool scheduled_thread_pool<Scheduler>::is_busy()
    {
        // If we are currently on a pika thread, which runs on the current pool,
        // we ignore it for the purposes of checking if the pool is busy (i.e.
        // this returns true only if there is *other* work left on this pool).
        std::int64_t pika_thread_offset =
            (threads::detail::get_self_ptr() && this_thread::get_pool() == this) ? 1 : 0;
        bool have_pika_threads =
            get_thread_count_unknown(std::size_t(-1), false) > pika_thread_offset;
        bool have_polling_work = sched_->Scheduler::get_polling_work_count() > 0;

        return have_pika_threads || have_polling_work;
    }

    template <typename Scheduler>
    bool scheduled_thread_pool<Scheduler>::is_idle()
    {
        return !is_busy();
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::wait()
    {
        pika::util::detail::yield_while_count(
            [this]() { return is_busy(); }, shutdown_check_count_, "scheduled_thread_pool::wait");
    }

    template <typename Scheduler>
    template <typename Lock>
    void scheduled_thread_pool<Scheduler>::stop_locked(Lock& l, bool blocking)
    {
        PIKA_LOG(info, "pool \"{}\" blocking({})", id_.name(), blocking);

        if (!threads_.empty())
        {
            // wait for all work to be done before requesting threads to shut
            // down
            if (blocking) { wait(); }

            // wake up if suspended
            resume_internal(blocking, throws);

            // set state to stopping
            sched_->Scheduler::set_all_states_at_least(runtime_state::stopping);

            // make sure we're not waiting
            sched_->Scheduler::do_some_work(std::size_t(-1));

            if (blocking)
            {
                for (std::size_t i = 0; i != threads_.size(); ++i)
                {
                    // skip this if already stopped
                    if (!threads_[i].joinable()) continue;

                    // make sure no OS thread is waiting
                    PIKA_LOG(info, "pool \"{}\" notify_all", id_.name());

                    sched_->Scheduler::do_some_work(std::size_t(-1));

                    PIKA_LOG(info, "pool \"{}\" join:{}", id_.name(), i);

                    {
                        // unlock the lock while joining
                        ::pika::detail::unlock_guard<Lock> ul(l);
                        remove_processing_unit_internal(i);
                    }
                }
                threads_.clear();
            }
        }
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::stop(std::unique_lock<std::mutex>& l, bool blocking)
    {
        PIKA_ASSERT(l.owns_lock());
        return stop_locked(l, blocking);
    }

    template <typename Scheduler>
    bool pika::threads::detail::scheduled_thread_pool<Scheduler>::run(
        std::unique_lock<std::mutex>& l, std::size_t pool_threads)
    {
        PIKA_ASSERT(l.owns_lock());

        PIKA_LOG(info, "pool \"{}\" number of processing units available: {}", id_.name(),
            threads::detail::hardware_concurrency());
        PIKA_LOG(info, "pool \"{}\" creating {} OS thread(s)", id_.name(), pool_threads);

        if (0 == pool_threads)
        {
            PIKA_THROW_EXCEPTION(pika::error::bad_parameter, "run", "number of threads is zero");
        }

        if (!threads_.empty() || sched_->Scheduler::has_reached_state(runtime_state::running))
        {
            return true;    // do nothing if already running
        }

        init_perf_counter_data(pool_threads);
        this->init_pool_time_scale();

        PIKA_LOG(info, "pool \"{}\" timestamp_scale: {}", id_.name(), timestamp_scale_);

        // run threads and wait for initialization to complete
        std::size_t thread_num = 0;
        std::shared_ptr<pika::concurrency::detail::barrier> startup =
            std::make_shared<pika::concurrency::detail::barrier>(pool_threads + 1);
        try
        {
            topology const& topo = get_topology();

            for (/**/; thread_num != pool_threads; ++thread_num)
            {
                std::size_t global_thread_num = this->thread_offset_ + thread_num;
                threads::detail::mask_cref_type mask =
                    affinity_data_.get_pu_mask(topo, global_thread_num);

                // thread_num ordering: 1. threads of default pool
                //                      2. threads of first special pool
                //                      3. etc.
                // get_pu_mask expects index according to ordering of masks
                // in affinity_data::affinity_masks_
                // which is in order of occupied PU
                PIKA_LOG(info,
                    "pool \"{}\" create OS thread {}: will run on processing units "
                    "within this mask: {}",
                    id_.name(), global_thread_num, pika::threads::detail::to_string(mask));

                // create a new thread
                add_processing_unit_internal(thread_num, global_thread_num, startup);
            }

            // wait for all threads to have started up
            startup->wait();

            PIKA_ASSERT(pool_threads == std::size_t(thread_count_.load()));
        }
        catch (std::exception const& e)
        {
            PIKA_LOG(critical, "pool \"{}\" failed with: {}", id_.name(), e.what());

            // trigger the barrier
            pool_threads -= (thread_num + 1);
            while (pool_threads-- != 0) startup->wait();

            stop_locked(l);
            threads_.clear();

            return false;
        }

        PIKA_LOG(info, "pool \"{}\" running", id_.name());
        return true;
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::resume_internal(bool blocking, error_code& ec)
    {
        for (std::size_t virt_core = 0; virt_core != threads_.size(); ++virt_core)
        {
            this->sched_->Scheduler::resume(virt_core);
        }

        if (blocking)
        {
            for (std::size_t virt_core = 0; virt_core != threads_.size(); ++virt_core)
            {
                if (threads_[virt_core].joinable())
                {
                    resume_processing_unit_direct(virt_core, ec);
                }
            }
        }
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::resume_direct(error_code& ec)
    {
        this->resume_internal(true, ec);
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::suspend_internal(error_code& ec)
    {
        util::yield_while([this]() { return this->sched_->Scheduler::get_thread_count() > 0; },
            "scheduled_thread_pool::suspend_internal");

        for (std::size_t i = 0; i != threads_.size(); ++i)
        {
            pika::runtime_state expected = runtime_state::running;
            sched_->Scheduler::get_state(i).compare_exchange_strong(
                expected, runtime_state::pre_sleep);
        }

        for (std::size_t i = 0; i != threads_.size(); ++i)
        {
            suspend_processing_unit_internal(i, ec);
        }
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::suspend_direct(error_code& ec)
    {
        if (threads::detail::get_self_ptr() && pika::this_thread::get_pool() == this)
        {
            PIKA_THROWS_IF(ec, pika::error::bad_parameter,
                "scheduled_thread_pool<Scheduler>::suspend_direct",
                "cannot suspend a pool from itself");
            return;
        }

        this->suspend_internal(ec);
    }

    template <typename Scheduler>
    void
    pika::threads::detail::scheduled_thread_pool<Scheduler>::thread_func(std::size_t thread_num,
        std::size_t global_thread_num, std::shared_ptr<pika::concurrency::detail::barrier> startup)
    {
        topology const& topo = get_topology();

        // Set the affinity for the current thread.
        threads::detail::mask_type mask = affinity_data_.get_pu_mask(topo, global_thread_num);

        // If the mask is empty (signaling no binding) we set a full mask anyway, because the whole
        // process may have a non-full mask set which is inherited by this thread.
        if (!any(mask)) { mask = topo.get_machine_affinity_mask(); }

        if (PIKA_LOG_ENABLED(debug)) topo.write_to_log();

        error_code ec(throwmode::lightweight);
        topo.set_thread_affinity_mask(mask, ec);
        if (ec)
        {
            PIKA_LOG(warn, "pool \"{}\" setting thread affinity on OS thread {} failed with: {}",
                id_.name(), global_thread_num, ec.get_message());
        }

        // Setting priority of worker threads to a lower priority, this
        // needs to
        // be done in order to give the parcel pool threads higher
        // priority
        if (get_scheduler()->has_scheduler_mode(scheduler_mode::reduce_thread_priority))
        {
            topo.reduce_thread_priority(ec);
            if (ec)
            {
                PIKA_LOG(warn,
                    "pool \"{}\" reducing thread priority on OS thread {} failed with: {}",
                    id_.name(), global_thread_num, ec.get_message());
            }
        }

        // manage the number of this thread in its TSS
        init_tss_helper<Scheduler> tss_helper(*this, thread_num, global_thread_num);

        ++thread_count_;

        // set state to running
        std::atomic<pika::runtime_state>& state = sched_->Scheduler::get_state(thread_num);
        [[maybe_unused]] pika::runtime_state oldstate = state.exchange(runtime_state::running);
        PIKA_ASSERT(oldstate <= runtime_state::running);

        // wait for all threads to start up before before starting pika work
        startup->wait();

        PIKA_LOG(info, "pool \"{}\" starting OS thread: {}", id_.name(), thread_num);

        try
        {
            try
            {
                manage_active_thread_count count(thread_count_);

                // run the work queue
                [[maybe_unused]] pika::threads::coroutines::detail::prepare_main_thread main_thread;

                // run main Scheduler loop until terminated
                scheduling_counter_data& counter_data = counter_data_[thread_num].data_;

                scheduling_counters counters(counter_data.executed_threads_,
                    counter_data.executed_thread_phases_, counter_data.tfunc_times_,
                    counter_data.exec_times_, counter_data.idle_loop_counts_,
                    counter_data.busy_loop_counts_, counter_data.tasks_active_);

                scheduling_callbacks callbacks(
                    util::detail::deferred_call(    //-V107
                        &scheduler_base::idle_callback, sched_.get(), thread_num),
                    nullptr, max_idle_loop_count_, max_busy_loop_count_);

                scheduling_loop(thread_num, *sched_, counters, callbacks);

                // the OS thread is allowed to exit only if no more pika
                // threads exist or if some other thread has terminated
                PIKA_ASSERT((sched_->Scheduler::get_thread_count(thread_schedule_state::suspended,
                                 execution::thread_priority::default_, thread_num) == 0 &&
                                sched_->Scheduler::get_queue_length(thread_num) == 0) ||
                    sched_->Scheduler::get_state(thread_num) > runtime_state::stopping);
            }
            catch (pika::exception const& e)
            {
                PIKA_LOG(critical,
                    "pool \"{}\" thread_num:{} : caught pika::exception: {}, "
                    "aborted thread execution",
                    id_.name(), global_thread_num, e.what());

                report_error(global_thread_num, std::current_exception());
                return;
            }
            catch (std::system_error const& e)
            {
                PIKA_LOG(critical,
                    "pool \"{}\" thread_num:{} : caught std::system_error: {}, "
                    "aborted thread execution",
                    id_.name(), global_thread_num, e.what());

                report_error(global_thread_num, std::current_exception());
                return;
            }
            catch (std::exception const& e)
            {
                // Repackage exceptions to avoid slicing.
                pika::throw_with_info(pika::exception(pika::error::unhandled_exception, e.what()));
            }
        }
        catch (...)
        {
            PIKA_LOG(critical,
                "pool \"{}\" thread_num:{} : caught unexpected exception, aborted "
                "thread execution",
                id_.name(), global_thread_num);

            report_error(global_thread_num, std::current_exception());
            return;
        }

        PIKA_LOG(info, "pool \"{}\" thread_num: {}, ending OS thread, executed {} pika threads",
            id_.name(), global_thread_num,
            counter_data_[global_thread_num].data_.executed_threads_);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::create_thread(
        thread_init_data& data, thread_id_ref_type& id, error_code& ec)
    {
        // verify state
        if (thread_count_ == 0 && !sched_->Scheduler::is_state(runtime_state::running))
        {
            // thread-manager is not currently running
            PIKA_THROWS_IF(ec, pika::error::invalid_status, "thread_pool<Scheduler>::create_thread",
                "invalid state: thread pool is not running");
            return;
        }

        threads::detail::create_thread(sched_.get(), data, id, ec);    //-V601
    }

    template <typename Scheduler>
    thread_id_ref_type
    scheduled_thread_pool<Scheduler>::create_work(thread_init_data& data, error_code& ec)
    {
        // verify state
        if (thread_count_ == 0 && !sched_->Scheduler::is_state(runtime_state::running))
        {
            // thread-manager is not currently running
            PIKA_THROWS_IF(ec, pika::error::invalid_status, "thread_pool<Scheduler>::create_work",
                "invalid state: thread pool is not running");
            return invalid_thread_id;
        }

        thread_id_ref_type id = threads::detail::create_work(sched_.get(), data, ec);    //-V601

        return id;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    thread_state scheduled_thread_pool<Scheduler>::set_state(thread_id_type const& id,
        thread_schedule_state new_state, thread_restart_state new_state_ex,
        execution::thread_priority priority, error_code& ec)
    {
        return set_thread_state(id, new_state,    //-V107
            new_state_ex, priority,
            execution::thread_schedule_hint(static_cast<std::int16_t>(get_local_thread_num_tss())),
            true, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    // performance counters
    template <typename InIter, typename OutIter, typename ProjSrc, typename ProjDest>
    OutIter
    copy_projected(InIter first, InIter last, OutIter dest, ProjSrc&& srcproj, ProjDest&& destproj)
    {
        while (first != last)
        {
            PIKA_INVOKE(destproj, (dest++)->data_) = PIKA_INVOKE(srcproj, (first++)->data_);
        }
        return dest;
    }

    template <typename InIter, typename T, typename Proj>
    T accumulate_projected(InIter first, InIter last, T init, Proj&& proj)
    {
        while (first != last) { init = std::move(init) + PIKA_INVOKE(proj, (first++)->data_); }
        return init;
    }

#if defined(PIKA_HAVE_THREAD_CUMULATIVE_COUNTS)
    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_executed_threads(std::size_t num, bool reset)
    {
        std::int64_t executed_threads = 0;
        std::int64_t reset_executed_threads = 0;

        if (num != std::size_t(-1))
        {
            executed_threads = counter_data_[num].data_.executed_threads_;
            reset_executed_threads = counter_data_[num].data_.reset_executed_threads_;

            if (reset) counter_data_[num].data_.reset_executed_threads_ = executed_threads;
        }
        else
        {
            executed_threads = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::executed_threads_);
            reset_executed_threads =
                accumulate_projected(counter_data_.begin(), counter_data_.end(), std::int64_t(0),
                    &scheduling_counter_data::reset_executed_threads_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                    &scheduling_counter_data::executed_threads_,
                    &scheduling_counter_data::reset_executed_threads_);
            }
        }

        PIKA_ASSERT(executed_threads >= reset_executed_threads);

        return executed_threads - reset_executed_threads;
    }
#endif

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_executed_threads() const
    {
        std::int64_t executed_threads = accumulate_projected(counter_data_.begin(),
            counter_data_.end(), std::int64_t(0), &scheduling_counter_data::executed_threads_);

#if defined(PIKA_HAVE_THREAD_CUMULATIVE_COUNTS)
        std::int64_t reset_executed_threads =
            accumulate_projected(counter_data_.begin(), counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_executed_threads_);

        PIKA_ASSERT(executed_threads >= reset_executed_threads);
        return executed_threads - reset_executed_threads;
#else
        return executed_threads;
#endif
    }

#if defined(PIKA_HAVE_THREAD_CUMULATIVE_COUNTS)
    template <typename Scheduler>
    std::int64_t
    scheduled_thread_pool<Scheduler>::get_executed_thread_phases(std::size_t num, bool reset)
    {
        std::int64_t executed_phases = 0;
        std::int64_t reset_executed_phases = 0;

        if (num != std::size_t(-1))
        {
            executed_phases = counter_data_[num].data_.executed_thread_phases_;
            reset_executed_phases = counter_data_[num].data_.reset_executed_thread_phases_;

            if (reset) counter_data_[num].data_.reset_executed_thread_phases_ = executed_phases;
        }
        else
        {
            executed_phases = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::executed_thread_phases_);
            reset_executed_phases = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::reset_executed_thread_phases_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                    &scheduling_counter_data::executed_thread_phases_,
                    &scheduling_counter_data::reset_executed_thread_phases_);
            }
        }

        PIKA_ASSERT(executed_phases >= reset_executed_phases);

        return executed_phases - reset_executed_phases;
    }

# if defined(PIKA_HAVE_THREAD_IDLE_RATES)
    template <typename Scheduler>
    std::int64_t
    scheduled_thread_pool<Scheduler>::get_thread_phase_duration(std::size_t num, bool reset)
    {
        std::int64_t exec_total = 0;
        std::int64_t num_phases = 0;
        std::int64_t reset_exec_total = 0;
        std::int64_t reset_num_phases = 0;

        if (num != std::size_t(-1))
        {
            exec_total = counter_data_[num].data_.exec_times_;
            num_phases = counter_data_[num].data_.executed_thread_phases_;

            reset_exec_total = counter_data_[num].data_.reset_thread_phase_duration_times_;
            reset_num_phases = counter_data_[num].data_.reset_thread_phase_duration_;

            if (reset)
            {
                counter_data_[num].data_.reset_thread_phase_duration_ = num_phases;
                counter_data_[num].data_.reset_thread_phase_duration_times_ = exec_total;
            }
        }
        else
        {
            exec_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::exec_times_);
            num_phases = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::executed_thread_phases_);

            reset_exec_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::reset_thread_phase_duration_times_);
            reset_num_phases = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::reset_thread_phase_duration_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                    &scheduling_counter_data::exec_times_,
                    &scheduling_counter_data::reset_thread_phase_duration_times_);
                copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                    &scheduling_counter_data::executed_thread_phases_,
                    &scheduling_counter_data::reset_thread_phase_duration_);
            }
        }

        PIKA_ASSERT(exec_total >= reset_exec_total);
        PIKA_ASSERT(num_phases >= reset_num_phases);

        exec_total -= reset_exec_total;
        num_phases -= reset_num_phases;

        return std::int64_t((double(exec_total) * timestamp_scale_) / double(num_phases));
    }

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_thread_duration(std::size_t num, bool reset)
    {
        std::int64_t exec_total = 0;
        std::int64_t num_threads = 0;
        std::int64_t reset_exec_total = 0;
        std::int64_t reset_num_threads = 0;

        if (num != std::size_t(-1))
        {
            exec_total = counter_data_[num].data_.exec_times_;
            num_threads = counter_data_[num].data_.executed_threads_;

            reset_exec_total = counter_data_[num].data_.reset_thread_duration_times_;
            reset_num_threads = counter_data_[num].data_.reset_thread_duration_;

            if (reset)
            {
                counter_data_[num].data_.reset_thread_duration_ = num_threads;
                counter_data_[num].data_.reset_thread_duration_times_ = exec_total;
            }
        }
        else
        {
            exec_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::exec_times_);
            num_threads = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::executed_threads_);

            reset_exec_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::reset_thread_duration_times_);
            reset_num_threads = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::reset_thread_duration_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                    &scheduling_counter_data::exec_times_,
                    &scheduling_counter_data::reset_thread_duration_times_);
                copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                    &scheduling_counter_data::executed_threads_,
                    &scheduling_counter_data::reset_thread_duration_);
            }
        }

        PIKA_ASSERT(exec_total >= reset_exec_total);
        PIKA_ASSERT(num_threads >= reset_num_threads);

        exec_total -= reset_exec_total;
        num_threads -= reset_num_threads;

        return std::int64_t((double(exec_total) * timestamp_scale_) / double(num_threads));
    }

    template <typename Scheduler>
    std::int64_t
    scheduled_thread_pool<Scheduler>::get_thread_phase_overhead(std::size_t num, bool reset)
    {
        std::int64_t exec_total = 0;
        std::int64_t tfunc_total = 0;
        std::int64_t num_phases = 0;

        std::int64_t reset_exec_total = 0;
        std::int64_t reset_tfunc_total = 0;
        std::int64_t reset_num_phases = 0;

        if (num != std::size_t(-1))
        {
            exec_total = counter_data_[num].data_.exec_times_;
            tfunc_total = counter_data_[num].data_.tfunc_times_;
            num_phases = counter_data_[num].data_.executed_thread_phases_;

            reset_exec_total = counter_data_[num].data_.reset_thread_phase_overhead_times_;
            reset_tfunc_total = counter_data_[num].data_.reset_thread_phase_overhead_times_total_;
            reset_num_phases = counter_data_[num].data_.reset_thread_phase_overhead_;

            if (reset)
            {
                counter_data_[num].data_.reset_thread_phase_overhead_times_ = exec_total;
                counter_data_[num].data_.reset_thread_phase_overhead_times_total_ = tfunc_total;
                counter_data_[num].data_.reset_thread_phase_overhead_ = num_phases;
            }
        }
        else
        {
            exec_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::exec_times_);
            tfunc_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::tfunc_times_);
            num_phases = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::executed_thread_phases_);

            reset_exec_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::reset_thread_phase_overhead_times_);
            reset_tfunc_total =
                accumulate_projected(counter_data_.begin(), counter_data_.end(), std::int64_t(0),
                    &scheduling_counter_data::reset_thread_phase_overhead_times_total_);
            reset_num_phases = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::reset_thread_phase_overhead_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                    &scheduling_counter_data::exec_times_,
                    &scheduling_counter_data::reset_thread_phase_overhead_times_);
                copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                    &scheduling_counter_data::tfunc_times_,
                    &scheduling_counter_data::reset_thread_phase_overhead_times_total_);
                copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                    &scheduling_counter_data::executed_thread_phases_,
                    &scheduling_counter_data::reset_thread_phase_overhead_);
            }
        }

        PIKA_ASSERT(exec_total >= reset_exec_total);
        PIKA_ASSERT(tfunc_total >= reset_tfunc_total);
        PIKA_ASSERT(num_phases >= reset_num_phases);

        exec_total -= reset_exec_total;
        tfunc_total -= reset_tfunc_total;
        num_phases -= reset_num_phases;

        if (num_phases == 0)    // avoid division by zero
            return 0;

        PIKA_ASSERT(tfunc_total >= exec_total);

        return std::int64_t(
            double((tfunc_total - exec_total) * timestamp_scale_) / double(num_phases));
    }

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_thread_overhead(std::size_t num, bool reset)
    {
        std::int64_t exec_total = 0;
        std::int64_t tfunc_total = 0;
        std::int64_t num_threads = 0;

        std::int64_t reset_exec_total = 0;
        std::int64_t reset_tfunc_total = 0;
        std::int64_t reset_num_threads = 0;

        if (num != std::size_t(-1))
        {
            exec_total = counter_data_[num].data_.exec_times_;
            tfunc_total = counter_data_[num].data_.tfunc_times_;
            num_threads = counter_data_[num].data_.executed_threads_;

            reset_exec_total = counter_data_[num].data_.reset_thread_overhead_times_;
            reset_tfunc_total = counter_data_[num].data_.reset_thread_overhead_times_total_;
            reset_num_threads = counter_data_[num].data_.reset_thread_overhead_;

            if (reset)
            {
                counter_data_[num].data_.reset_thread_overhead_times_ = exec_total;
                counter_data_[num].data_.reset_thread_overhead_times_total_ = tfunc_total;
                counter_data_[num].data_.reset_thread_overhead_ = num_threads;
            }
        }
        else
        {
            exec_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::exec_times_);
            tfunc_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::tfunc_times_);
            num_threads = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::executed_threads_);

            reset_exec_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::reset_thread_overhead_times_);
            reset_tfunc_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::reset_thread_overhead_times_total_);
            reset_num_threads = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::reset_thread_overhead_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                    &scheduling_counter_data::exec_times_,
                    &scheduling_counter_data::reset_thread_overhead_times_);
                copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                    &scheduling_counter_data::tfunc_times_,
                    &scheduling_counter_data::reset_thread_overhead_times_total_);
                copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                    &scheduling_counter_data::executed_thread_phases_,
                    &scheduling_counter_data::reset_thread_overhead_);
            }
        }

        PIKA_ASSERT(exec_total >= reset_exec_total);
        PIKA_ASSERT(tfunc_total >= reset_tfunc_total);
        PIKA_ASSERT(num_threads >= reset_num_threads);

        exec_total -= reset_exec_total;
        tfunc_total -= reset_tfunc_total;
        num_threads -= reset_num_threads;

        if (num_threads == 0)    // avoid division by zero
            return 0;

        PIKA_ASSERT(tfunc_total >= exec_total);

        return std::int64_t(
            double((tfunc_total - exec_total) * timestamp_scale_) / double(num_threads));
    }

    template <typename Scheduler>
    std::int64_t
    scheduled_thread_pool<Scheduler>::get_cumulative_thread_duration(std::size_t num, bool reset)
    {
        std::int64_t exec_total = 0;
        std::int64_t reset_exec_total = 0;

        if (num != std::size_t(-1))
        {
            exec_total = counter_data_[num].data_.exec_times_;
            reset_exec_total = counter_data_[num].data_.reset_cumulative_thread_duration_;

            if (reset) { counter_data_[num].data_.reset_cumulative_thread_duration_ = exec_total; }
        }
        else
        {
            exec_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::exec_times_);
            reset_exec_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::reset_cumulative_thread_duration_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                    &scheduling_counter_data::exec_times_,
                    &scheduling_counter_data::reset_cumulative_thread_duration_);
            }
        }

        PIKA_ASSERT(exec_total >= reset_exec_total);

        exec_total -= reset_exec_total;

        return std::int64_t(double(exec_total) * timestamp_scale_);
    }

    template <typename Scheduler>
    std::int64_t
    scheduled_thread_pool<Scheduler>::get_cumulative_thread_overhead(std::size_t num, bool reset)
    {
        std::int64_t exec_total = 0;
        std::int64_t reset_exec_total = 0;
        std::int64_t tfunc_total = 0;
        std::int64_t reset_tfunc_total = 0;

        if (num != std::size_t(-1))
        {
            exec_total = counter_data_[num].data_.exec_times_;
            tfunc_total = counter_data_[num].data_.tfunc_times_;

            reset_exec_total = counter_data_[num].data_.reset_cumulative_thread_overhead_;
            reset_tfunc_total = counter_data_[num].data_.reset_cumulative_thread_overhead_total_;

            if (reset)
            {
                counter_data_[num].data_.reset_cumulative_thread_overhead_ = exec_total;
                counter_data_[num].data_.reset_cumulative_thread_overhead_total_ = tfunc_total;
            }
        }
        else
        {
            exec_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::exec_times_);
            reset_exec_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::reset_cumulative_thread_overhead_);

            tfunc_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::tfunc_times_);
            reset_tfunc_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::reset_cumulative_thread_overhead_total_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                    &scheduling_counter_data::exec_times_,
                    &scheduling_counter_data::reset_cumulative_thread_overhead_);
                copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                    &scheduling_counter_data::tfunc_times_,
                    &scheduling_counter_data::reset_cumulative_thread_overhead_total_);
            }
        }

        PIKA_ASSERT(exec_total >= reset_exec_total);
        PIKA_ASSERT(tfunc_total >= reset_tfunc_total);

        exec_total -= reset_exec_total;
        tfunc_total -= reset_tfunc_total;

        return std::int64_t((double(tfunc_total) - double(exec_total)) * timestamp_scale_);
    }
# endif    // PIKA_HAVE_THREAD_IDLE_RATES
#endif     // PIKA_HAVE_THREAD_CUMULATIVE_COUNTS

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    std::int64_t
    scheduled_thread_pool<Scheduler>::get_cumulative_duration(std::size_t num, bool reset)
    {
        std::int64_t tfunc_total = 0;
        std::int64_t reset_tfunc_total = 0;

        if (num != std::size_t(-1))
        {
            tfunc_total = counter_data_[num].data_.tfunc_times_;
            reset_tfunc_total = counter_data_[num].data_.reset_tfunc_times_;

            if (reset) counter_data_[num].data_.reset_tfunc_times_ = tfunc_total;
        }
        else
        {
            tfunc_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::tfunc_times_);
            reset_tfunc_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::reset_tfunc_times_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                    &scheduling_counter_data::tfunc_times_,
                    &scheduling_counter_data::reset_tfunc_times_);
            }
        }

        PIKA_ASSERT(tfunc_total >= reset_tfunc_total);

        tfunc_total -= reset_tfunc_total;

        return std::int64_t(double(tfunc_total) * timestamp_scale_);
    }

#if defined(PIKA_HAVE_THREAD_IDLE_RATES)
# if defined(PIKA_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::avg_creation_idle_rate(std::size_t, bool reset)
    {
        double const creation_total =
            static_cast<double>(sched_->Scheduler::get_creation_time(reset));

        std::int64_t exec_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
            std::int64_t(0), &scheduling_counter_data::exec_times_);
        std::int64_t tfunc_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
            std::int64_t(0), &scheduling_counter_data::tfunc_times_);

        std::int64_t reset_exec_total =
            accumulate_projected(counter_data_.begin(), counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_creation_idle_rate_time_);
        std::int64_t reset_tfunc_total =
            accumulate_projected(counter_data_.begin(), counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_creation_idle_rate_time_total_);

        if (reset)
        {
            copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                &scheduling_counter_data::exec_times_,
                &scheduling_counter_data::reset_creation_idle_rate_time_);
            copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                &scheduling_counter_data::tfunc_times_,
                &scheduling_counter_data::reset_creation_idle_rate_time_);
        }

        PIKA_ASSERT(exec_total >= reset_exec_total);
        PIKA_ASSERT(tfunc_total >= reset_tfunc_total);

        exec_total -= reset_exec_total;
        tfunc_total -= reset_tfunc_total;

        if (tfunc_total == exec_total)    // avoid division by zero
            return 10000LL;

        PIKA_ASSERT(tfunc_total > exec_total);

        double const percent = (creation_total / double(tfunc_total - exec_total));
        return std::int64_t(10000. * percent);    // 0.01 percent
    }

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::avg_cleanup_idle_rate(std::size_t, bool reset)
    {
        double const cleanup_total =
            static_cast<double>(sched_->Scheduler::get_cleanup_time(reset));

        std::int64_t exec_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
            std::int64_t(0), &scheduling_counter_data::exec_times_);
        std::int64_t tfunc_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
            std::int64_t(0), &scheduling_counter_data::tfunc_times_);

        std::int64_t reset_exec_total =
            accumulate_projected(counter_data_.begin(), counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_cleanup_idle_rate_time_);
        std::int64_t reset_tfunc_total =
            accumulate_projected(counter_data_.begin(), counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_cleanup_idle_rate_time_total_);

        if (reset)
        {
            copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                &scheduling_counter_data::exec_times_,
                &scheduling_counter_data::reset_cleanup_idle_rate_time_);
            copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                &scheduling_counter_data::tfunc_times_,
                &scheduling_counter_data::reset_cleanup_idle_rate_time_);
        }

        PIKA_ASSERT(exec_total >= reset_exec_total);
        PIKA_ASSERT(tfunc_total >= reset_tfunc_total);

        exec_total -= reset_exec_total;
        tfunc_total -= reset_tfunc_total;

        if (tfunc_total == exec_total)    // avoid division by zero
            return 10000LL;

        PIKA_ASSERT(tfunc_total > exec_total);

        double const percent = (cleanup_total / double(tfunc_total - exec_total));
        return std::int64_t(10000. * percent);    // 0.01 percent
    }
# endif    // PIKA_HAVE_THREAD_CREATION_AND_CLEANUP_RATES

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::avg_idle_rate_all(bool reset)
    {
        std::int64_t exec_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
            std::int64_t(0), &scheduling_counter_data::exec_times_);
        std::int64_t tfunc_total = accumulate_projected(counter_data_.begin(), counter_data_.end(),
            std::int64_t(0), &scheduling_counter_data::tfunc_times_);

        std::int64_t reset_exec_total = accumulate_projected(counter_data_.begin(),
            counter_data_.end(), std::int64_t(0), &scheduling_counter_data::reset_idle_rate_time_);
        std::int64_t reset_tfunc_total =
            accumulate_projected(counter_data_.begin(), counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_idle_rate_time_total_);

        if (reset)
        {
            copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                &scheduling_counter_data::exec_times_,
                &scheduling_counter_data::reset_idle_rate_time_);
            copy_projected(counter_data_.begin(), counter_data_.end(), counter_data_.begin(),
                &scheduling_counter_data::tfunc_times_,
                &scheduling_counter_data::reset_idle_rate_time_total_);
        }

        PIKA_ASSERT(exec_total >= reset_exec_total);
        PIKA_ASSERT(tfunc_total >= reset_tfunc_total);

        exec_total -= reset_exec_total;
        tfunc_total -= reset_tfunc_total;

        if (tfunc_total == 0)    // avoid division by zero
            return 10000LL;

        PIKA_ASSERT(tfunc_total >= exec_total);

        double const percent = 1. - (double(exec_total) / double(tfunc_total));
        return std::int64_t(10000. * percent);    // 0.01 percent
    }

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::avg_idle_rate(std::size_t num, bool reset)
    {
        if (num == std::size_t(-1)) return avg_idle_rate_all(reset);

        std::int64_t exec_time = counter_data_[num].data_.exec_times_;
        std::int64_t tfunc_time = counter_data_[num].data_.tfunc_times_;
        std::int64_t reset_exec_time = counter_data_[num].data_.reset_idle_rate_time_;
        std::int64_t reset_tfunc_time = counter_data_[num].data_.reset_idle_rate_time_total_;

        if (reset)
        {
            counter_data_[num].data_.reset_idle_rate_time_ = exec_time;
            counter_data_[num].data_.reset_idle_rate_time_total_ = tfunc_time;
        }

        PIKA_ASSERT(exec_time >= reset_exec_time);
        PIKA_ASSERT(tfunc_time >= reset_tfunc_time);

        exec_time -= reset_exec_time;
        tfunc_time -= reset_tfunc_time;

        if (tfunc_time == 0)    // avoid division by zero
            return 10000LL;

        PIKA_ASSERT(tfunc_time > exec_time);

        double const percent = 1. - (double(exec_time) / double(tfunc_time));
        return std::int64_t(10000. * percent);    // 0.01 percent
    }
#endif    // PIKA_HAVE_THREAD_IDLE_RATES

    template <typename Scheduler>
    std::int64_t
    scheduled_thread_pool<Scheduler>::get_idle_loop_count(std::size_t num, bool /* reset */)
    {
        if (num == std::size_t(-1))
        {
            return accumulate_projected(counter_data_.begin(), counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::idle_loop_counts_);
        }
        return counter_data_[num].data_.idle_loop_counts_;
    }

    template <typename Scheduler>
    std::int64_t
    scheduled_thread_pool<Scheduler>::get_busy_loop_count(std::size_t num, bool /* reset */)
    {
        if (num == std::size_t(-1))
        {
            return accumulate_projected(counter_data_.begin(), counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::busy_loop_counts_);
        }
        return counter_data_[num].data_.busy_loop_counts_;
    }

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_scheduler_utilization() const
    {
        return (accumulate_projected(counter_data_.begin(), counter_data_.end(), std::int64_t(0),
                    &scheduling_counter_data::tasks_active_) *
                   100) /
            thread_count_.load();
    }

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_idle_core_count() const
    {
        std::int64_t count = 0;
        std::size_t i = 0;
        for (auto const& data : counter_data_)
        {
            if (!data.data_.tasks_active_ && sched_->Scheduler::is_core_idle(i)) { ++count; }
            ++i;
        }
        return count;
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::get_idle_core_mask(mask_type& mask) const
    {
        std::size_t i = 0;
        for (auto const& data : counter_data_)
        {
            if (!data.data_.tasks_active_ && sched_->Scheduler::is_core_idle(i)) { set(mask, i); }
            ++i;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::init_perf_counter_data(std::size_t pool_threads)
    {
        counter_data_.resize(pool_threads);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::add_processing_unit_internal(std::size_t virt_core,
        std::size_t thread_num, std::shared_ptr<pika::concurrency::detail::barrier> startup,
        error_code& ec)
    {
        std::unique_lock<typename Scheduler::pu_mutex_type> l(
            sched_->Scheduler::get_pu_mutex(virt_core));

        if (threads_.size() <= virt_core) threads_.resize(virt_core + 1);

        if (threads_[virt_core].joinable())
        {
            l.unlock();
            PIKA_THROWS_IF(ec, pika::error::bad_parameter,
                "scheduled_thread_pool<Scheduler>::add_processing_unit",
                "the given virtual core has already been added to this thread pool");
            return;
        }

        std::atomic<pika::runtime_state>& state = sched_->Scheduler::get_state(virt_core);
        [[maybe_unused]] pika::runtime_state oldstate = state.exchange(runtime_state::initialized);
        PIKA_ASSERT(oldstate == runtime_state::stopped || oldstate == runtime_state::initialized);

        threads_[virt_core] = std::thread(
            &scheduled_thread_pool::thread_func, this, virt_core, thread_num, std::move(startup));

        if (&ec != &throws) ec = make_success_code();
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::remove_processing_unit_internal(
        std::size_t virt_core, error_code& ec)
    {
        std::unique_lock<typename Scheduler::pu_mutex_type> l(
            sched_->Scheduler::get_pu_mutex(virt_core));

        if (threads_.size() <= virt_core || !threads_[virt_core].joinable())
        {
            l.unlock();
            PIKA_THROWS_IF(ec, pika::error::bad_parameter,
                "scheduled_thread_pool<Scheduler>::remove_processing_unit",
                "the given virtual core has already been stopped to run on this thread pool");
            return;
        }

        std::atomic<pika::runtime_state>& state = sched_->Scheduler::get_state(virt_core);

        // inform the scheduler to stop the virtual core
        pika::runtime_state oldstate = state.exchange(runtime_state::stopping);

        if (oldstate > runtime_state::stopping)
        {
            // If thread was terminating or already stopped we don't want to
            // change the value back to stopping, so we restore the old state.
            state.store(oldstate);
        }

        PIKA_ASSERT(oldstate == runtime_state::starting || oldstate == runtime_state::running ||
            oldstate == runtime_state::stopping || oldstate == runtime_state::stopped ||
            oldstate == runtime_state::terminating);

        std::thread t;
        std::swap(threads_[virt_core], t);

        l.unlock();

        if (threads::detail::get_self_ptr() && this == pika::this_thread::get_pool())
        {
            std::size_t thread_num = thread_offset_ + virt_core;

            util::yield_while(
                [thread_num]() { return thread_num == pika::get_worker_thread_num(); },
                "scheduled_thread_pool::remove_processing_unit_internal");
        }

        t.join();
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::suspend_processing_unit_internal(
        std::size_t virt_core, error_code& ec)
    {
        // Yield to other pika threads if lock is not available to avoid
        // deadlocks when multiple pika threads try to resume or suspend pus.
        std::unique_lock<typename Scheduler::pu_mutex_type> l(
            sched_->Scheduler::get_pu_mutex(virt_core), std::defer_lock);

        util::yield_while([&l]() { return !l.try_lock(); },
            "scheduled_thread_pool::suspend_processing_unit_internal");

        if (threads_.size() <= virt_core || !threads_[virt_core].joinable())
        {
            l.unlock();
            PIKA_THROWS_IF(ec, pika::error::bad_parameter,
                "scheduled_thread_pool<Scheduler>::suspend_processing_unit_internal",
                "the given virtual core has already been stopped to run on this thread pool");
            return;
        }

        std::atomic<pika::runtime_state>& state = sched_->Scheduler::get_state(virt_core);

        // Inform the scheduler to suspend the virtual core only if running
        pika::runtime_state expected = runtime_state::running;
        state.compare_exchange_strong(expected, runtime_state::pre_sleep);

        l.unlock();

        PIKA_ASSERT(expected == runtime_state::running || expected == runtime_state::pre_sleep ||
            expected == runtime_state::sleeping);

        util::yield_while([&state]() { return state.load() == runtime_state::pre_sleep; },
            "scheduled_thread_pool::suspend_processing_unit_internal");
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::suspend_processing_unit_direct(
        std::size_t virt_core, error_code& ec)
    {
        if (!get_scheduler()->has_scheduler_mode(scheduler_mode::enable_elasticity))
        {
            PIKA_THROWS_IF(ec, pika::error::invalid_status,
                "scheduled_thread_pool<Scheduler>::suspend_processing_unit_direct",
                "this thread pool does not support suspending processing units");
        }

        if (threads::detail::get_self_ptr() &&
            !get_scheduler()->has_scheduler_mode(scheduler_mode::enable_stealing) &&
            pika::this_thread::get_pool() == this)
        {
            PIKA_THROWS_IF(ec, pika::error::invalid_status,
                "scheduled_thread_pool<Scheduler>::suspend_processing_unit_direct",
                "this thread pool does not support suspending processing units from itself (no "
                "thread stealing)");
        }

        suspend_processing_unit_internal(virt_core, ec);
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::resume_processing_unit_direct(
        std::size_t virt_core, error_code& ec)
    {
        // Yield to other pika threads if lock is not available to avoid
        // deadlocks when multiple pika threads try to resume or suspend pus.
        std::unique_lock<typename Scheduler::pu_mutex_type> l(
            sched_->Scheduler::get_pu_mutex(virt_core), std::defer_lock);
        util::yield_while([&l]() { return !l.try_lock(); },
            "scheduled_thread_pool::resume_processing_unit_direct");

        if (threads_.size() <= virt_core || !threads_[virt_core].joinable())
        {
            l.unlock();
            PIKA_THROWS_IF(ec, pika::error::bad_parameter,
                "scheduled_thread_pool<Scheduler>::resume_processing_unit",
                "the given virtual core has already been stopped to run on this thread pool");
            return;
        }

        l.unlock();

        std::atomic<pika::runtime_state>& state = sched_->Scheduler::get_state(virt_core);

        util::yield_while(
            [this, &state, virt_core]() {
                this->sched_->Scheduler::resume(virt_core);
                return state.load() == runtime_state::sleeping;
            },
            "scheduled_thread_pool::resume_processing_unit_direct");
    }
}    // namespace pika::threads::detail
