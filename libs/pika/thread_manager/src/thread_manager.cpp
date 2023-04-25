//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach, Katelyn Kufahl
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2015 Patricia Grubel
//  Copyright (c) 2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/async_combinators/wait_all.hpp>
#include <pika/execution_base/this_thread.hpp>
#include <pika/functional/bind.hpp>
#include <pika/futures/future.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/logging.hpp>
#include <pika/modules/schedulers.hpp>
#include <pika/modules/thread_manager.hpp>
#include <pika/resource_partitioner/detail/partitioner.hpp>
#include <pika/runtime_configuration/runtime_configuration.hpp>
#include <pika/thread_pool_util/thread_pool_suspension_helpers.hpp>
#include <pika/thread_pools/scheduled_thread_pool.hpp>
#include <pika/threading_base/set_thread_state.hpp>
#include <pika/threading_base/thread_data.hpp>
#include <pika/threading_base/thread_helpers.hpp>
#include <pika/threading_base/thread_init_data.hpp>
#include <pika/threading_base/thread_queue_init_parameters.hpp>
#include <pika/timing/detail/timestamp.hpp>
#include <pika/topology/topology.hpp>
#include <pika/type_support/unused.hpp>
#include <pika/util/get_entry_as.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace pika::threads::detail {
    void check_num_high_priority_queues(
        std::size_t num_threads, std::size_t num_high_priority_queues)
    {
        if (num_high_priority_queues > num_threads)
        {
            throw pika::detail::command_line_error(
                "Invalid command line option: number of high priority threads "
                "(--pika:high-priority-threads), should not be larger than number of threads "
                "(--pika:threads)");
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_manager::thread_manager(
        pika::util::runtime_configuration& rtcfg, notification_policy_type& notifier)
      : rtcfg_(rtcfg)
      , notifier_(notifier)
    {
        using std::placeholders::_1;
        using std::placeholders::_3;

        // Add callbacks local to thread_manager.
        notifier.add_on_start_thread_callback(
            util::detail::bind(&thread_manager::init_tss, this, _1));
        notifier.add_on_stop_thread_callback(util::detail::bind(&thread_manager::deinit_tss, this));

        auto& rp = pika::resource::get_partitioner();
        notifier.add_on_start_thread_callback(
            util::detail::bind(&resource::detail::partitioner::assign_pu, std::ref(rp), _3, _1));
        notifier.add_on_stop_thread_callback(
            util::detail::bind(&resource::detail::partitioner::unassign_pu, std::ref(rp), _3, _1));
    }

    void thread_manager::create_pools()
    {
        auto& rp = pika::resource::get_partitioner();
        size_t num_pools = rp.get_num_pools();
        std::size_t thread_offset = 0;

        std::size_t const max_idle_loop_count = pika::detail::get_entry_as<std::int64_t>(
            rtcfg_, "pika.max_idle_loop_count", PIKA_IDLE_LOOP_COUNT_MAX);
        std::size_t const max_busy_loop_count = pika::detail::get_entry_as<std::int64_t>(
            rtcfg_, "pika.max_busy_loop_count", PIKA_BUSY_LOOP_COUNT_MAX);

        std::int64_t const max_thread_count = pika::detail::get_entry_as<std::int64_t>(
            rtcfg_, "pika.thread_queue.max_thread_count", PIKA_THREAD_QUEUE_MAX_THREAD_COUNT);
        std::int64_t const min_tasks_to_steal_pending = pika::detail::get_entry_as<std::int64_t>(
            rtcfg_, "pika.thread_queue.min_tasks_to_steal_pending",
            PIKA_THREAD_QUEUE_MIN_TASKS_TO_STEAL_PENDING);
        std::int64_t const min_tasks_to_steal_staged = pika::detail::get_entry_as<std::int64_t>(
            rtcfg_, "pika.thread_queue.min_tasks_to_steal_staged",
            PIKA_THREAD_QUEUE_MIN_TASKS_TO_STEAL_STAGED);
        std::int64_t const min_add_new_count = pika::detail::get_entry_as<std::int64_t>(
            rtcfg_, "pika.thread_queue.min_add_new_count", PIKA_THREAD_QUEUE_MIN_ADD_NEW_COUNT);
        std::int64_t const max_add_new_count = pika::detail::get_entry_as<std::int64_t>(
            rtcfg_, "pika.thread_queue.max_add_new_count", PIKA_THREAD_QUEUE_MAX_ADD_NEW_COUNT);
        std::int64_t const min_delete_count = pika::detail::get_entry_as<std::int64_t>(
            rtcfg_, "pika.thread_queue.min_delete_count", PIKA_THREAD_QUEUE_MIN_DELETE_COUNT);
        std::int64_t const max_delete_count = pika::detail::get_entry_as<std::int64_t>(
            rtcfg_, "pika.thread_queue.max_delete_count", PIKA_THREAD_QUEUE_MAX_DELETE_COUNT);
        std::int64_t const max_terminated_threads = pika::detail::get_entry_as<std::int64_t>(rtcfg_,
            "pika.thread_queue.max_terminated_threads", PIKA_THREAD_QUEUE_MAX_TERMINATED_THREADS);
        std::int64_t const init_threads_count = pika::detail::get_entry_as<std::int64_t>(
            rtcfg_, "pika.thread_queue.init_threads_count", PIKA_THREAD_QUEUE_INIT_THREADS_COUNT);
        double const max_idle_backoff_time = pika::detail::get_entry_as<double>(
            rtcfg_, "pika.max_idle_backoff_time", PIKA_IDLE_BACKOFF_TIME_MAX);

        std::ptrdiff_t small_stacksize = rtcfg_.get_stack_size(execution::thread_stacksize::small_);
        std::ptrdiff_t medium_stacksize =
            rtcfg_.get_stack_size(execution::thread_stacksize::medium);
        std::ptrdiff_t large_stacksize = rtcfg_.get_stack_size(execution::thread_stacksize::large);
        std::ptrdiff_t huge_stacksize = rtcfg_.get_stack_size(execution::thread_stacksize::huge);

        thread_queue_init_parameters thread_queue_init(max_thread_count, min_tasks_to_steal_pending,
            min_tasks_to_steal_staged, min_add_new_count, max_add_new_count, min_delete_count,
            max_delete_count, max_terminated_threads, init_threads_count, max_idle_backoff_time,
            small_stacksize, medium_stacksize, large_stacksize, huge_stacksize);

        // instantiate the pools
        for (size_t i = 0; i != num_pools; i++)
        {
            std::string name = rp.get_pool_name(i);
            resource::scheduling_policy sched_type = rp.which_scheduler(name);
            std::size_t num_threads_in_pool = rp.get_num_threads(i);
            scheduler_mode scheduler_mode = rp.get_scheduler_mode(i);

            // make sure the first thread-pool that gets instantiated is the default one
            if (i == 0)
            {
                if (name != rp.get_default_pool_name())
                {
                    throw std::invalid_argument("Trying to instantiate pool " + name +
                        " as first thread pool, but first thread pool must be named " +
                        rp.get_default_pool_name());
                }
            }

            thread_pool_init_parameters thread_pool_init(name, i, scheduler_mode,
                num_threads_in_pool, thread_offset, notifier_, rp.get_affinity_data(),
                max_idle_loop_count, max_busy_loop_count);

            std::size_t numa_sensitive =
                pika::detail::get_entry_as<std::size_t>(rtcfg_, "pika.numa_sensitive", 0);

            switch (sched_type)
            {
            case resource::user_defined:
            {
                auto pool_func = rp.get_pool_creator(i);
                std::unique_ptr<thread_pool_base> pool(
                    pool_func(thread_pool_init, thread_queue_init));
                pools_.push_back(PIKA_MOVE(pool));
                break;
            }
            case resource::unspecified:
            {
                throw std::invalid_argument(
                    "cannot instantiate a thread-manager if the thread-pool" + name +
                    " has an unspecified scheduler type");
            }
            case resource::local:
            {
                // instantiate the scheduler
                using local_sched_type = pika::threads::detail::local_queue_scheduler<>;

                local_sched_type::init_parameter_type init(thread_pool_init.num_threads_,
                    thread_pool_init.affinity_data_, thread_queue_init,
                    "core-local_queue_scheduler");

                std::unique_ptr<local_sched_type> sched(new local_sched_type(init));

                // set the default scheduler flags
                sched->set_scheduler_mode(thread_pool_init.mode_);
                // conditionally set/unset this flag
                sched->update_scheduler_mode(scheduler_mode::enable_stealing_numa, !numa_sensitive);

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new pika::threads::detail::scheduled_thread_pool<local_sched_type>(
                        PIKA_MOVE(sched), thread_pool_init));
                pools_.push_back(PIKA_MOVE(pool));
                break;
            }

            case resource::local_priority_fifo:
            {
                // set parameters for scheduler and pool instantiation and
                // perform compatibility checks
                std::size_t num_high_priority_queues =
                    pika::detail::get_entry_as<std::size_t>(rtcfg_,
                        "pika.thread_queue.high_priority_queues", thread_pool_init.num_threads_);
                check_num_high_priority_queues(
                    thread_pool_init.num_threads_, num_high_priority_queues);

                // instantiate the scheduler
                using local_sched_type =
                    pika::threads::detail::local_priority_queue_scheduler<std::mutex,
                        pika::threads::detail::lockfree_fifo>;

                local_sched_type::init_parameter_type init(thread_pool_init.num_threads_,
                    thread_pool_init.affinity_data_, num_high_priority_queues, thread_queue_init,
                    "core-local_priority_queue_scheduler");

                std::unique_ptr<local_sched_type> sched(new local_sched_type(init));

                // set the default scheduler flags
                sched->set_scheduler_mode(thread_pool_init.mode_);
                // conditionally set/unset this flag
                sched->update_scheduler_mode(scheduler_mode::enable_stealing_numa, !numa_sensitive);

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new pika::threads::detail::scheduled_thread_pool<local_sched_type>(
                        PIKA_MOVE(sched), thread_pool_init));
                pools_.push_back(PIKA_MOVE(pool));

                break;
            }

            case resource::local_priority_lifo:
            {
#if defined(PIKA_HAVE_CXX11_STD_ATOMIC_128BIT)
                // set parameters for scheduler and pool instantiation and
                // perform compatibility checks
                std::size_t num_high_priority_queues =
                    pika::detail::get_entry_as<std::size_t>(rtcfg_,
                        "pika.thread_queue.high_priority_queues", thread_pool_init.num_threads_);
                check_num_high_priority_queues(
                    thread_pool_init.num_threads_, num_high_priority_queues);

                // instantiate the scheduler
                using local_sched_type =
                    pika::threads::detail::local_priority_queue_scheduler<std::mutex,
                        pika::threads::detail::lockfree_lifo>;

                local_sched_type::init_parameter_type init(thread_pool_init.num_threads_,
                    thread_pool_init.affinity_data_, num_high_priority_queues, thread_queue_init,
                    "core-local_priority_queue_scheduler");

                std::unique_ptr<local_sched_type> sched(new local_sched_type(init));

                // set the default scheduler flags
                sched->set_scheduler_mode(thread_pool_init.mode_);
                // conditionally set/unset this flag
                sched->update_scheduler_mode(scheduler_mode::enable_stealing_numa, !numa_sensitive);

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new pika::threads::detail::scheduled_thread_pool<local_sched_type>(
                        PIKA_MOVE(sched), thread_pool_init));
                pools_.push_back(PIKA_MOVE(pool));
#else
                throw pika::detail::command_line_error(
                    "Command line option --pika:queuing=local-priority-lifo is not configured in "
                    "this build. Please make sure 128bit atomics are available.");
#endif
                break;
            }

            case resource::static_:
            {
                // instantiate the scheduler
                using local_sched_type = pika::threads::detail::static_queue_scheduler<>;

                local_sched_type::init_parameter_type init(thread_pool_init.num_threads_,
                    thread_pool_init.affinity_data_, thread_queue_init,
                    "core-static_queue_scheduler");

                std::unique_ptr<local_sched_type> sched(new local_sched_type(init));

                // set the default scheduler flags
                sched->set_scheduler_mode(thread_pool_init.mode_);
                // conditionally set/unset this flag
                sched->update_scheduler_mode(scheduler_mode::enable_stealing_numa, !numa_sensitive);

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new pika::threads::detail::scheduled_thread_pool<local_sched_type>(
                        PIKA_MOVE(sched), thread_pool_init));
                pools_.push_back(PIKA_MOVE(pool));
                break;
            }

            case resource::static_priority:
            {
                // set parameters for scheduler and pool instantiation and
                // perform compatibility checks
                std::size_t num_high_priority_queues =
                    pika::detail::get_entry_as<std::size_t>(rtcfg_,
                        "pika.thread_queue.high_priority_queues", thread_pool_init.num_threads_);
                check_num_high_priority_queues(
                    thread_pool_init.num_threads_, num_high_priority_queues);

                // instantiate the scheduler
                using local_sched_type = pika::threads::detail::static_priority_queue_scheduler<>;

                local_sched_type::init_parameter_type init(thread_pool_init.num_threads_,
                    thread_pool_init.affinity_data_, num_high_priority_queues, thread_queue_init,
                    "core-static_priority_queue_scheduler");

                std::unique_ptr<local_sched_type> sched(new local_sched_type(init));

                // set the default scheduler flags
                sched->set_scheduler_mode(thread_pool_init.mode_);
                // conditionally set/unset this flag
                sched->update_scheduler_mode(scheduler_mode::enable_stealing_numa, !numa_sensitive);

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new pika::threads::detail::scheduled_thread_pool<local_sched_type>(
                        PIKA_MOVE(sched), thread_pool_init));
                pools_.push_back(PIKA_MOVE(pool));
                break;
            }

            case resource::abp_priority_fifo:
            {
#if defined(PIKA_HAVE_CXX11_STD_ATOMIC_128BIT)
                // set parameters for scheduler and pool instantiation and
                // perform compatibility checks
                std::size_t num_high_priority_queues =
                    pika::detail::get_entry_as<std::size_t>(rtcfg_,
                        "pika.thread_queue.high_priority_queues", thread_pool_init.num_threads_);
                check_num_high_priority_queues(
                    thread_pool_init.num_threads_, num_high_priority_queues);

                // instantiate the scheduler
                using local_sched_type =
                    pika::threads::detail::local_priority_queue_scheduler<std::mutex,
                        pika::threads::detail::lockfree_fifo>;

                local_sched_type::init_parameter_type init(thread_pool_init.num_threads_,
                    thread_pool_init.affinity_data_, num_high_priority_queues, thread_queue_init,
                    "core-abp_fifo_priority_queue_scheduler");

                std::unique_ptr<local_sched_type> sched(new local_sched_type(init));

                // set the default scheduler flags
                sched->set_scheduler_mode(thread_pool_init.mode_);
                // conditionally set/unset this flag
                sched->update_scheduler_mode(scheduler_mode::enable_stealing_numa, !numa_sensitive);

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new pika::threads::detail::scheduled_thread_pool<local_sched_type>(
                        PIKA_MOVE(sched), thread_pool_init));
                pools_.push_back(PIKA_MOVE(pool));
#else
                throw pika::detail::command_line_error(
                    "Command line option --pika:queuing=abp-priority-fifo is not configured in "
                    "this build. Please make sure 128bit atomics are available.");
#endif
                break;
            }

            case resource::abp_priority_lifo:
            {
#if defined(PIKA_HAVE_CXX11_STD_ATOMIC_128BIT)
                // set parameters for scheduler and pool instantiation and
                // perform compatibility checks
                std::size_t num_high_priority_queues =
                    pika::detail::get_entry_as<std::size_t>(rtcfg_,
                        "pika.thread_queue.high_priority_queues", thread_pool_init.num_threads_);
                check_num_high_priority_queues(
                    thread_pool_init.num_threads_, num_high_priority_queues);

                // instantiate the scheduler
                using local_sched_type =
                    pika::threads::detail::local_priority_queue_scheduler<std::mutex,
                        pika::threads::detail::lockfree_lifo>;

                local_sched_type::init_parameter_type init(thread_pool_init.num_threads_,
                    thread_pool_init.affinity_data_, num_high_priority_queues, thread_queue_init,
                    "core-abp_fifo_priority_queue_scheduler");

                std::unique_ptr<local_sched_type> sched(new local_sched_type(init));

                // set the default scheduler flags
                sched->set_scheduler_mode(thread_pool_init.mode_);
                // conditionally set/unset this flag
                sched->update_scheduler_mode(scheduler_mode::enable_stealing_numa, !numa_sensitive);

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new pika::threads::detail::scheduled_thread_pool<local_sched_type>(
                        PIKA_MOVE(sched), thread_pool_init));
                pools_.push_back(PIKA_MOVE(pool));
#else
                throw pika::detail::command_line_error(
                    "Command line option --pika:queuing=abp-priority-lifo is not configured in "
                    "this build. Please make sure 128bit atomics are available.");
#endif
                break;
            }

            case resource::shared_priority:
            {
                // instantiate the scheduler
                using local_sched_type = pika::threads::detail::shared_priority_queue_scheduler<>;
                local_sched_type::init_parameter_type init(thread_pool_init.num_threads_, {1, 1, 1},
                    thread_pool_init.affinity_data_, thread_queue_init,
                    "core-shared_priority_queue_scheduler");

                std::unique_ptr<local_sched_type> sched(new local_sched_type(init));

                // set the default scheduler flags
                sched->set_scheduler_mode(thread_pool_init.mode_);
                // conditionally set/unset this flag
                sched->update_scheduler_mode(scheduler_mode::enable_stealing_numa, !numa_sensitive);

                // instantiate the pool
                std::unique_ptr<thread_pool_base> pool(
                    new pika::threads::detail::scheduled_thread_pool<local_sched_type>(
                        PIKA_MOVE(sched), thread_pool_init));
                pools_.push_back(PIKA_MOVE(pool));
                break;
            }
            }

            // update the thread_offset for the next pool
            thread_offset += num_threads_in_pool;
        }

        // fill the thread-lookup table
        for (auto& pool_iter : pools_)
        {
            std::size_t nt = rp.get_num_threads(pool_iter->get_pool_index());
            for (std::size_t i = 0; i < nt; i++)
            {
                threads_lookup_.push_back(pool_iter->get_pool_id());
            }
        }
    }

    thread_manager::~thread_manager() {}

    void thread_manager::init()
    {
        auto& rp = pika::resource::get_partitioner();
        std::size_t threads_offset = 0;

        // initialize all pools
        for (auto&& pool_iter : pools_)
        {
            std::size_t num_threads_in_pool = rp.get_num_threads(pool_iter->get_pool_index());
            pool_iter->init(num_threads_in_pool, threads_offset);
            threads_offset += num_threads_in_pool;
        }
    }

    void thread_manager::print_pools(std::ostream& os)
    {
        os << "The thread-manager owns " << pools_.size()    //  -V128
           << " pool(s) : \n";

        for (auto&& pool_iter : pools_)
        {
            pool_iter->print_pool(os);
        }
    }

    thread_pool_base& thread_manager::default_pool() const
    {
        PIKA_ASSERT(!pools_.empty());
        return *pools_[0];
    }

    thread_pool_base& thread_manager::get_pool(std::string const& pool_name) const
    {
        // if the given pool_name is default, we don't need to look for it
        // we must always return pool 0
        if (pool_name == "default" ||
            pool_name == resource::get_partitioner().get_default_pool_name())
        {
            return default_pool();
        }

        // now check the other pools - no need to check pool 0 again, so ++begin
        auto pool = std::find_if(
            ++pools_.begin(), pools_.end(), [&pool_name](pool_type const& itp) -> bool {
                return (itp->get_pool_name() == pool_name);
            });

        if (pool != pools_.end())
        {
            return **pool;
        }

        //! FIXME Add names of available pools?
        PIKA_THROW_EXCEPTION(pika::error::bad_parameter, "thread_manager::get_pool",
            "the resource partitioner does not own a thread pool named '{}'.\n", pool_name);
    }

    thread_pool_base& thread_manager::get_pool(pool_id_type const& pool_id) const
    {
        return get_pool(pool_id.name());
    }

    thread_pool_base& thread_manager::get_pool(std::size_t thread_index) const
    {
        return get_pool(threads_lookup_[thread_index]);
    }

    bool thread_manager::pool_exists(std::string const& pool_name) const
    {
        // if the given pool_name is default, we don't need to look for it
        // we must always return pool 0
        if (pool_name == "default" ||
            pool_name == resource::get_partitioner().get_default_pool_name())
        {
            return true;
        }

        // now check the other pools - no need to check pool 0 again, so ++begin
        auto pool = std::find_if(
            ++pools_.begin(), pools_.end(), [&pool_name](pool_type const& itp) -> bool {
                return (itp->get_pool_name() == pool_name);
            });

        if (pool != pools_.end())
        {
            return true;
        }

        return false;
    }

    bool thread_manager::pool_exists(std::size_t pool_index) const
    {
        return pool_index < pools_.size();
    }

    ///////////////////////////////////////////////////////////////////////////
    std::int64_t thread_manager::get_thread_count(thread_schedule_state state,
        execution::thread_priority priority, std::size_t num_thread, bool reset)
    {
        std::int64_t total_count = 0;
        std::lock_guard<mutex_type> lk(mtx_);

        for (auto& pool_iter : pools_)
        {
            total_count += pool_iter->get_thread_count(state, priority, num_thread, reset);
        }

        return total_count;
    }

    std::int64_t thread_manager::get_idle_core_count()
    {
        std::int64_t total_count = 0;
        std::lock_guard<mutex_type> lk(mtx_);

        for (auto& pool_iter : pools_)
        {
            total_count += pool_iter->get_idle_core_count();
        }

        return total_count;
    }

    mask_type thread_manager::get_idle_core_mask()
    {
        mask_type mask = mask_type();
        resize(mask, hardware_concurrency());

        std::lock_guard<mutex_type> lk(mtx_);

        for (auto& pool_iter : pools_)
        {
            pool_iter->get_idle_core_mask(mask);
        }

        return mask;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Enumerate all matching threads
    bool thread_manager::enumerate_threads(
        util::detail::function<bool(thread_id_type)> const& f, thread_schedule_state state) const
    {
        std::lock_guard<mutex_type> lk(mtx_);
        bool result = true;

        for (auto& pool_iter : pools_)
        {
            result = result && pool_iter->enumerate_threads(f, state);
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Abort all threads which are in suspended state. This will set
    // the state of all suspended threads to \a pending while
    // supplying the wait_abort extended state flag
    void thread_manager::abort_all_suspended_threads()
    {
        std::lock_guard<mutex_type> lk(mtx_);
        for (auto& pool_iter : pools_)
        {
            pool_iter->abort_all_suspended_threads();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Clean up terminated threads. This deletes all threads which
    // have been terminated but which are still held in the queue
    // of terminated threads. Some schedulers might not do anything
    // here.
    bool thread_manager::cleanup_terminated(bool delete_all)
    {
        std::lock_guard<mutex_type> lk(mtx_);
        bool result = true;

        for (auto& pool_iter : pools_)
        {
            result = pool_iter->cleanup_terminated(delete_all) && result;
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    void thread_manager::register_thread(
        thread_init_data& data, thread_id_ref_type& id, error_code& ec)
    {
        thread_pool_base* pool = nullptr;
        auto thrd_data = get_self_id_data();
        if (thrd_data)
        {
            pool = thrd_data->get_scheduler_base()->get_parent_pool();
        }
        else
        {
            pool = &default_pool();
        }
        pool->create_thread(data, id, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_id_ref_type thread_manager::register_work(thread_init_data& data, error_code& ec)
    {
        thread_pool_base* pool = nullptr;
        auto thrd_data = get_self_id_data();
        if (thrd_data)
        {
            pool = thrd_data->get_scheduler_base()->get_parent_pool();
        }
        else
        {
            pool = &default_pool();
        }
        return pool->create_work(data, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    constexpr std::size_t all_threads = std::size_t(-1);

    std::int64_t thread_manager::get_queue_length(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_queue_length(all_threads, reset);
        return result;
    }

#ifdef PIKA_HAVE_THREAD_QUEUE_WAITTIME
    std::int64_t thread_manager::get_average_thread_wait_time(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_average_thread_wait_time(all_threads, reset);
        return result;
    }

    std::int64_t thread_manager::get_average_task_wait_time(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_average_task_wait_time(all_threads, reset);
        return result;
    }
#endif

    std::int64_t thread_manager::get_cumulative_duration(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_cumulative_duration(all_threads, reset);
        return result;
    }

#ifdef PIKA_HAVE_THREAD_IDLE_RATES
    std::int64_t thread_manager::avg_idle_rate(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->avg_idle_rate(all_threads, reset);
        return result;
    }

# ifdef PIKA_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
    std::int64_t thread_manager::avg_creation_idle_rate(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->avg_creation_idle_rate(all_threads, reset);
        return result;
    }

    std::int64_t thread_manager::avg_cleanup_idle_rate(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->avg_cleanup_idle_rate(all_threads, reset);
        return result;
    }
# endif
#endif

#ifdef PIKA_HAVE_THREAD_CUMULATIVE_COUNTS
    std::int64_t thread_manager::get_executed_threads(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_executed_threads(all_threads, reset);
        return result;
    }

    std::int64_t thread_manager::get_executed_thread_phases(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_executed_thread_phases(all_threads, reset);
        return result;
    }

# ifdef PIKA_HAVE_THREAD_IDLE_RATES
    std::int64_t thread_manager::get_thread_duration(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_thread_duration(all_threads, reset);
        return result;
    }

    std::int64_t thread_manager::get_thread_phase_duration(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_thread_phase_duration(all_threads, reset);
        return result;
    }

    std::int64_t thread_manager::get_thread_overhead(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_thread_overhead(all_threads, reset);
        return result;
    }

    std::int64_t thread_manager::get_thread_phase_overhead(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_thread_phase_overhead(all_threads, reset);
        return result;
    }

    std::int64_t thread_manager::get_cumulative_thread_duration(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_cumulative_thread_duration(all_threads, reset);
        return result;
    }

    std::int64_t thread_manager::get_cumulative_thread_overhead(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_cumulative_thread_overhead(all_threads, reset);
        return result;
    }
# endif
#endif

#ifdef PIKA_HAVE_THREAD_STEALING_COUNTS
    std::int64_t thread_manager::get_num_pending_misses(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_num_pending_misses(all_threads, reset);
        return result;
    }

    std::int64_t thread_manager::get_num_pending_accesses(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_num_pending_accesses(all_threads, reset);
        return result;
    }

    std::int64_t thread_manager::get_num_stolen_from_pending(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_num_stolen_from_pending(all_threads, reset);
        return result;
    }

    std::int64_t thread_manager::get_num_stolen_from_staged(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_num_stolen_from_staged(all_threads, reset);
        return result;
    }

    std::int64_t thread_manager::get_num_stolen_to_pending(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_num_stolen_to_pending(all_threads, reset);
        return result;
    }

    std::int64_t thread_manager::get_num_stolen_to_staged(bool reset)
    {
        std::int64_t result = 0;
        for (auto const& pool_iter : pools_)
            result += pool_iter->get_num_stolen_to_staged(all_threads, reset);
        return result;
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    bool thread_manager::run()
    {
        std::unique_lock<mutex_type> lk(mtx_);

        // the main thread needs to have a unique thread_num
        // worker threads are numbered 0..N-1, so we can use N for this thread
        auto& rp = pika::resource::get_partitioner();
        init_tss(rp.get_num_threads());

        for (auto& pool_iter : pools_)
        {
            std::size_t num_threads_in_pool = rp.get_num_threads(pool_iter->get_pool_name());

            if (pool_iter->get_os_thread_count() != 0 ||
                pool_iter->has_reached_state(runtime_state::running))
            {
                return true;    // do nothing if already running
            }

            if (!pool_iter->run(lk, num_threads_in_pool))
            {
                return false;
            }

            // set all states of all schedulers to "running"
            scheduler_base* sched = pool_iter->get_scheduler();
            if (sched)
                sched->set_all_states(runtime_state::running);
        }

        LTM_(info).format("run: running");
        return true;
    }

    void thread_manager::stop(bool blocking)
    {
        LTM_(info).format("stop: blocking({})", blocking ? "true" : "false");

        std::unique_lock<mutex_type> lk(mtx_);
        for (auto& pool_iter : pools_)
        {
            pool_iter->stop(lk, blocking);
        }
        deinit_tss();
    }

    bool thread_manager::is_busy()
    {
        bool busy = false;
        for (auto& pool_iter : pools_)
        {
            busy = busy || pool_iter->is_busy();
        }
        return busy;
    }

    bool thread_manager::is_idle()
    {
        bool idle = true;
        for (auto& pool_iter : pools_)
        {
            idle = idle && pool_iter->is_idle();
        }
        return idle;
    }

    void thread_manager::wait()
    {
        std::size_t shutdown_check_count =
            ::pika::detail::get_entry_as<std::size_t>(rtcfg_, "pika.shutdown_check_count", 10);
        pika::util::detail::yield_while_count([this]() { return is_busy(); }, shutdown_check_count);
    }

    void thread_manager::suspend()
    {
        wait();

        if (threads::detail::get_self_ptr())
        {
            std::vector<pika::future<void>> fs;

            for (auto& pool_iter : pools_)
            {
                fs.push_back(suspend_pool(*pool_iter));
            }

            pika::wait_all(fs);
        }
        else
        {
            for (auto& pool_iter : pools_)
            {
                pool_iter->suspend_direct();
            }
        }
    }

    void thread_manager::resume()
    {
        if (threads::detail::get_self_ptr())
        {
            std::vector<pika::future<void>> fs;

            for (auto& pool_iter : pools_)
            {
                fs.push_back(resume_pool(*pool_iter));
            }
            pika::wait_all(fs);
        }
        else
        {
            for (auto& pool_iter : pools_)
            {
                pool_iter->resume_direct();
            }
        }
    }
}    // namespace pika::threads::detail
