//  Copyright (c) 2007-2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/execution_base/this_thread.hpp>
#include <pika/threading_base/scheduler_base.hpp>
#include <pika/threading_base/scheduler_mode.hpp>
#include <pika/threading_base/scheduler_state.hpp>
#include <pika/threading_base/thread_init_data.hpp>
#include <pika/threading_base/thread_pool_base.hpp>
#if defined(PIKA_HAVE_SCHEDULER_LOCAL_STORAGE)
# include <pika/coroutines/detail/tss.hpp>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace pika::threads::detail {
    scheduler_base::scheduler_base(std::size_t num_threads, char const* description,
        thread_queue_init_parameters thread_queue_init, scheduler_mode mode)
      : suspend_mtxs_(num_threads)
      , suspend_conds_(num_threads)
      , pu_mtxs_(num_threads)
      , states_(num_threads)
      , description_(description)
      , thread_queue_init_(thread_queue_init)
      , parent_pool_(nullptr)
      , polling_function_mpi_(&null_polling_function)
      , polling_function_cuda_(&null_polling_function)
      , polling_work_count_function_mpi_(&null_polling_work_count_function)
      , polling_work_count_function_cuda_(&null_polling_work_count_function)
    {
        set_scheduler_mode(mode);

#if defined(PIKA_HAVE_THREAD_MANAGER_IDLE_BACKOFF)
        double max_time = thread_queue_init.max_idle_backoff_time_;

        wait_counts_.resize(num_threads);
        for (auto&& data : wait_counts_)
        {
            data.data_.wait_count_ = 0;
            data.data_.max_idle_backoff_time_ = max_time;
        }
#endif

        for (std::size_t i = 0; i != num_threads; ++i)
            states_[i].store(runtime_state::initialized);
    }

    void scheduler_base::idle_callback(std::size_t num_thread)
    {
#if defined(PIKA_HAVE_THREAD_MANAGER_IDLE_BACKOFF)
        if (has_scheduler_mode(scheduler_mode::enable_idle_backoff))
        {
            // Put this thread to sleep for some time, additionally it gets
            // woken up on new work.

            idle_backoff_data& data = wait_counts_[num_thread].data_;

            // Exponential back-off with a maximum sleep time.
            double exponent = (std::min)(
                double(data.wait_count_), double(std::numeric_limits<double>::max_exponent - 1));

            std::chrono::milliseconds period(
                std::lround((std::min)(data.max_idle_backoff_time_, std::pow(2.0, exponent))));

            ++data.wait_count_;

            std::unique_lock<pu_mutex_type> l(mtx_);
            if (cond_.wait_for(l, period) == std::cv_status::no_timeout)
            {
                // reset counter if thread was woken up
                data.wait_count_ = 0;
            }
        }
#else
        (void) num_thread;
#endif
    }

    /// This function gets called by the thread-manager whenever new work
    /// has been added, allowing the scheduler to reactivate one or more of
    /// possibly idling OS threads
    void scheduler_base::do_some_work(std::size_t)
    {
#if defined(PIKA_HAVE_THREAD_MANAGER_IDLE_BACKOFF)
        if (has_scheduler_mode(scheduler_mode::enable_idle_backoff))
        {
            cond_.notify_all();
        }
#endif
    }

    void scheduler_base::suspend(std::size_t num_thread)
    {
        PIKA_ASSERT(num_thread < suspend_conds_.size());

        states_[num_thread].store(runtime_state::sleeping);
        std::unique_lock<pu_mutex_type> l(suspend_mtxs_[num_thread]);
        suspend_conds_[num_thread].wait(l);

        // Only set running if still in runtime_state::sleeping. Can be set with
        // non-blocking/locking functions to stopping or terminating, in
        // which case the state is left untouched.
        pika::runtime_state expected = runtime_state::sleeping;
        states_[num_thread].compare_exchange_strong(expected, runtime_state::running);

        PIKA_ASSERT(expected == runtime_state::sleeping || expected == runtime_state::stopping ||
            expected == runtime_state::terminating);
    }

    void scheduler_base::resume(std::size_t num_thread)
    {
        if (num_thread == std::size_t(-1))
        {
            for (std::condition_variable& c : suspend_conds_)
            {
                c.notify_one();
            }
        }
        else
        {
            PIKA_ASSERT(num_thread < suspend_conds_.size());
            suspend_conds_[num_thread].notify_one();
        }
    }

    std::size_t scheduler_base::select_active_pu(
        std::unique_lock<pu_mutex_type>& l, std::size_t num_thread, bool allow_fallback)
    {
        if (has_scheduler_mode(threads::scheduler_mode::enable_elasticity))
        {
            std::size_t states_size = states_.size();

            if (!allow_fallback)
            {
                // Try indefinitely as long as at least one thread is
                // available for scheduling. Increase allowed state if no
                // threads are available for scheduling.
                auto max_allowed_state = runtime_state::suspended;

                pika::util::yield_while([this, states_size, &l, &num_thread, &max_allowed_state]() {
                    std::size_t num_allowed_threads = 0;

                    for (std::size_t offset = 0; offset < states_size; ++offset)
                    {
                        std::size_t num_thread_local = (num_thread + offset) % states_size;

                        l = std::unique_lock<pu_mutex_type>(
                            pu_mtxs_[num_thread_local], std::try_to_lock);

                        if (l.owns_lock())
                        {
                            if (states_[num_thread_local] <= max_allowed_state)
                            {
                                num_thread = num_thread_local;
                                return false;
                            }

                            l.unlock();
                        }

                        if (states_[num_thread_local] <= max_allowed_state)
                        {
                            ++num_allowed_threads;
                        }
                    }

                    if (0 == num_allowed_threads)
                    {
                        if (max_allowed_state <= runtime_state::suspended)
                        {
                            max_allowed_state = runtime_state::sleeping;
                        }
                        else if (max_allowed_state <= runtime_state::sleeping)
                        {
                            max_allowed_state = runtime_state::stopping;
                        }
                        else
                        {
                            // All threads are terminating or stopped.
                            // Just return num_thread to avoid infinite
                            // loop.
                            return false;
                        }
                    }

                    // Yield after trying all pus, then try again
                    return true;
                });

                return num_thread;
            }

            // Try all pus only once if fallback is allowed
            PIKA_ASSERT(num_thread != std::size_t(-1));
            for (std::size_t offset = 0; offset < states_size; ++offset)
            {
                std::size_t num_thread_local = (num_thread + offset) % states_size;

                l = std::unique_lock<pu_mutex_type>(pu_mtxs_[num_thread_local], std::try_to_lock);

                if (l.owns_lock() && states_[num_thread_local] <= runtime_state::suspended)
                {
                    return num_thread_local;
                }
            }
        }

        return num_thread;
    }

    // allow to access/manipulate states
    std::atomic<pika::runtime_state>& scheduler_base::get_state(std::size_t num_thread)
    {
        PIKA_ASSERT(num_thread < states_.size());
        return states_[num_thread];
    }
    std::atomic<pika::runtime_state> const& scheduler_base::get_state(std::size_t num_thread) const
    {
        PIKA_ASSERT(num_thread < states_.size());
        return states_[num_thread];
    }

    void scheduler_base::set_all_states(pika::runtime_state s)
    {
        using state_type = std::atomic<pika::runtime_state>;
        for (state_type& state : states_)
        {
            state.store(s);
        }
    }

    void scheduler_base::set_all_states_at_least(pika::runtime_state s)
    {
        using state_type = std::atomic<pika::runtime_state>;
        for (state_type& state : states_)
        {
            if (state < s)
            {
                state.store(s);
            }
        }
    }

    // return whether all states are at least at the given one
    bool scheduler_base::has_reached_state(pika::runtime_state s) const
    {
        using state_type = std::atomic<pika::runtime_state>;
        for (state_type const& state : states_)
        {
            if (state.load(std::memory_order_relaxed) < s)
                return false;
        }
        return true;
    }

    bool scheduler_base::is_state(pika::runtime_state s) const
    {
        using state_type = std::atomic<pika::runtime_state>;
        for (state_type const& state : states_)
        {
            if (state.load(std::memory_order_relaxed) != s)
                return false;
        }
        return true;
    }

    std::pair<pika::runtime_state, pika::runtime_state> scheduler_base::get_minmax_state() const
    {
        std::pair<pika::runtime_state, pika::runtime_state> result(
            runtime_state::last_valid_runtime, runtime_state::first_valid_runtime);

        using state_type = std::atomic<pika::runtime_state>;
        for (state_type const& state_iter : states_)
        {
            pika::runtime_state s = state_iter.load();
            result.first = (std::min)(result.first, s);
            result.second = (std::max)(result.second, s);
        }

        return result;
    }

    // get/set scheduler mode
    void scheduler_base::set_scheduler_mode(scheduler_mode mode)
    {
        // distribute the same value across all cores
        mode_.data_.store(mode, std::memory_order_release);
        do_some_work(std::size_t(-1));
    }

    void scheduler_base::add_scheduler_mode(scheduler_mode mode)
    {
        // distribute the same value across all cores
        mode = scheduler_mode(get_scheduler_mode() | mode);
        set_scheduler_mode(mode);
    }

    void scheduler_base::remove_scheduler_mode(scheduler_mode mode)
    {
        mode = scheduler_mode(get_scheduler_mode() & ~mode);
        set_scheduler_mode(mode);
    }

    void scheduler_base::update_scheduler_mode(scheduler_mode mode, bool set)
    {
        if (set)
        {
            add_scheduler_mode(mode);
        }
        else
        {
            remove_scheduler_mode(mode);
        }
    }

#if defined(PIKA_HAVE_SCHEDULER_LOCAL_STORAGE)
    coroutines::detail::tss_data_node* scheduler_base::find_tss_data(void const* key)
    {
        if (!thread_data_)
            return nullptr;
        return thread_data_->find(key);
    }

    void scheduler_base::add_new_tss_node(void const* key,
        std::shared_ptr<coroutines::detail::tss_cleanup_function> const& func, void* tss_data)
    {
        if (!thread_data_)
        {
            thread_data_ = std::make_shared<coroutines::detail::tss_storage>();
        }
        thread_data_->insert(key, func, tss_data);
    }

    void scheduler_base::erase_tss_node(void const* key, bool cleanup_existing)
    {
        if (thread_data_)
            thread_data_->erase(key, cleanup_existing);
    }

    void* scheduler_base::get_tss_data(void const* key)
    {
        if (coroutines::detail::tss_data_node* const current_node = find_tss_data(key))
        {
            return current_node->get_value();
        }
        return nullptr;
    }

    void scheduler_base::set_tss_data(void const* key,
        std::shared_ptr<coroutines::detail::tss_cleanup_function> const& func, void* tss_data,
        bool cleanup_existing)
    {
        if (coroutines::detail::tss_data_node* const current_node = find_tss_data(key))
        {
            if (func || (tss_data != 0))
                current_node->reinit(func, tss_data, cleanup_existing);
            else
                erase_tss_node(key, cleanup_existing);
        }
        else if (func || (tss_data != 0))
        {
            add_new_tss_node(key, func, tss_data);
        }
    }
#endif
}    // namespace pika::threads::detail
