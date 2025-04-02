//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/allocator_support/internal_allocator.hpp>
#include <pika/assert.hpp>
#include <pika/concurrency/cache_line_data.hpp>
#include <pika/functional/function.hpp>
#include <pika/modules/errors.hpp>
#include <pika/schedulers/deadlock_detection.hpp>
#include <pika/schedulers/lockfree_queue_backends.hpp>
#include <pika/schedulers/maintain_queue_wait_times.hpp>
#include <pika/schedulers/queue_holder_thread.hpp>
#include <pika/schedulers/thread_queue.hpp>
#include <pika/thread_support/unlock_guard.hpp>
#include <pika/threading_base/thread_data.hpp>
#include <pika/threading_base/thread_queue_init_parameters.hpp>
#include <pika/topology/topology.hpp>
#include <pika/util/get_and_reset_value.hpp>

#ifdef PIKA_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
# include <pika/timing/tick_counter.hpp>
#endif

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pika::detail {
    static pika::debug::detail::enable_print<false> tqmc_deb("_TQ_MC_");
}

///////////////////////////////////////////////////////////////////////////////
namespace pika::threads::detail {

    class thread_queue_mc
    {
    public:
        using thread_queue_type = thread_queue_mc;

        using thread_heap_type = std::list<threads::detail::thread_id_type,
            pika::detail::internal_allocator<threads::detail::thread_id_type>>;

        using task_description = threads::detail::thread_init_data;
        using thread_description = threads::detail::thread_data;

        using work_items_type = lockfree_fifo::apply<threads::detail::thread_id_ref_type>::type;
        using task_items_type = lockfree_fifo::apply<task_description>::type;

    public:
        // ----------------------------------------------------------------
        // Take thread init data from the new work queue and convert it into
        // full thread_data items that are added to the pending queue.
        //
        // New work items are taken from the queue owned by 'addfrom' and
        // added to the pending queue of this thread holder
        //
        // This is not thread safe, only the thread owning the holder should
        // call this function
        std::size_t add_new(std::int64_t add_count, thread_queue_type* addfrom, bool stealing)
        {
            [[maybe_unused]] auto scp = ::pika::detail::tqmc_deb.scope(debug::detail::ptr(this),
                __func__, "from", debug::detail::ptr(addfrom), "std::thread::id",
                debug::detail::hex<6>(holder_->owner_id_), stealing);
            PIKA_ASSERT(holder_->owner_id_ == std::this_thread::get_id());

            if (addfrom->new_tasks_count_.data_.load(std::memory_order_relaxed) == 0) { return 0; }

            std::size_t added = 0;
            task_description task;
            while (add_count-- && addfrom->new_task_items_.pop(task, stealing))
            {
                // create the new thread
                threads::detail::thread_init_data& data = task;
                threads::detail::thread_id_ref_type tid;

                holder_->create_thread_object(tid, data);
                holder_->add_to_thread_map(tid.noref());

                // Decrement only after thread_map_count_ has been incremented
                --addfrom->new_tasks_count_.data_;

                ::pika::detail::tqmc_deb.debug(debug::detail::str<>("add_new"), "stealing",
                    stealing,
                    debug::detail::threadinfo<threads::detail::thread_id_ref_type*>(&tid));

                // insert the thread into work-items queue assuming it is in
                // pending state
                PIKA_ASSERT(data.initial_state == threads::detail::thread_schedule_state::pending);

                // pushing the new thread into the pending queue of the
                // specified thread_queue
                ++added;
                schedule_work(std::move(tid), stealing);
            }

            return added;
        }

    public:
        explicit thread_queue_mc(
            thread_queue_init_parameters const& parameters, std::size_t queue_num = std::size_t(-1))
          : parameters_(parameters)
          , queue_index_(static_cast<int>(queue_num))
          , holder_(nullptr)
          , new_task_items_(1024)
          , work_items_(1024)
        {
            new_tasks_count_.data_ = 0;
            work_items_count_.data_ = 0;
        }

        // ----------------------------------------------------------------
        void set_holder(queue_holder_thread<thread_queue_type>* holder)
        {
            holder_ = holder;
            ::pika::detail::tqmc_deb.debug(debug::detail::str<>("set_holder"), "D",
                debug::detail::dec<2>(holder_->domain_index_), "Q",
                debug::detail::dec<3>(queue_index_));
        }

        // ----------------------------------------------------------------
        ~thread_queue_mc() {}

        // ----------------------------------------------------------------
        // This returns the current length of the queues (work items and new items)
        std::int64_t get_queue_length() const
        {
            return work_items_count_.data_.load(std::memory_order_relaxed) +
                new_tasks_count_.data_.load(std::memory_order_relaxed);
        }

        // ----------------------------------------------------------------
        // This returns the current length of the pending queue
        std::int64_t get_queue_length_pending() const
        {
            return work_items_count_.data_.load(std::memory_order_relaxed);
        }

        // ----------------------------------------------------------------
        // This returns the current length of the staged queue
        std::int64_t get_queue_length_staged(
            std::memory_order order = std::memory_order_relaxed) const
        {
            return new_tasks_count_.data_.load(order);
        }

        // ----------------------------------------------------------------
        // Return the number of existing threads with the given state.
        std::int64_t get_thread_count() const
        {
            PIKA_THROW_EXCEPTION(pika::error::bad_parameter, "get_thread_count",
                "use get_queue_length_staged/get_queue_length_pending");
            return 0;
        }

        // create a new thread and schedule it if the initial state is equal to
        // pending
        void create_thread(threads::detail::thread_init_data& data,
            threads::detail::thread_id_ref_type* id, error_code& ec)
        {
            // thread has not been created yet
            if (id) *id = threads::detail::invalid_thread_id;

            if (data.stacksize == execution::thread_stacksize::current)
            {
                data.stacksize = threads::detail::get_self_stacksize_enum();
            }

            PIKA_ASSERT(data.stacksize != execution::thread_stacksize::current);

            if (data.run_now)
            {
                threads::detail::thread_id_ref_type tid;
                holder_->create_thread_object(tid, data);
                holder_->add_to_thread_map(tid.noref());

                // push the new thread in the pending queue thread
                if (data.initial_state == threads::detail::thread_schedule_state::pending)
                {
                    // return the thread_id_ref of the newly created thread
                    if (id) { *id = tid; }
                    schedule_work(std::move(tid), false);
                }
                else
                {
                    // if the thread should not be scheduled the id must be
                    // returned to the caller as otherwise the thread would
                    // go out of scope right away.
                    PIKA_ASSERT(id != nullptr);
                    *id = std::move(tid);
                }

                if (&ec != &throws) ec = make_success_code();
                return;
            }

            // if the initial state is not pending, delayed creation will
            // fail as the newly created thread would go out of scope right
            // away (can't be scheduled).
            if (data.initial_state != threads::detail::thread_schedule_state::pending)
            {
                PIKA_THROW_EXCEPTION(pika::error::bad_parameter, "thread_queue_mc::create_thread",
                    "staged tasks must have 'pending' as their initial state");
            }

            // do not execute the work, but register a task description for
            // later thread creation
            ++new_tasks_count_.data_;

            new_task_items_.push(task_description(std::move(data)));

            if (&ec != &throws) ec = make_success_code();
        }

        // ----------------------------------------------------------------
        /// Return the next thread to be executed,
        /// return false if none is available
        bool get_next_thread(threads::detail::thread_id_ref_type& thrd, bool other_end,
            bool check_new = false) PIKA_HOT
        {
            [[maybe_unused]] auto scp =
                ::pika::detail::tqmc_deb.scope(debug::detail::ptr(this), __func__);
            std::int64_t work_items_count_count =
                work_items_count_.data_.load(std::memory_order_relaxed);

            // If there is an available thread on the work queue
            if (0 != work_items_count_count && work_items_.pop(thrd, other_end))
            {
                --work_items_count_.data_;
                ::pika::detail::tqmc_deb.debug(debug::detail::str<>("get_next_thread"), "stealing",
                    other_end, "D", debug::detail::dec<2>(holder_->domain_index_), "Q",
                    debug::detail::dec<3>(queue_index_), "n",
                    debug::detail::dec<4>(new_tasks_count_.data_), "w",
                    debug::detail::dec<4>(work_items_count_.data_),
                    debug::detail::threadinfo<threads::detail::thread_id_ref_type*>(&thrd));
                return true;
            }

            // if there is not any work ready, convert ready tasks into threads
            // not that if other_end is true = stealing, so do not convert
            // for thread safety reasons
            if (!other_end && check_new && add_new(32, this, false) > 0)
            {
                // use check_new false to prevent infinite recursion
                return get_next_thread(thrd, other_end, false);
            }
            return false;
        }

        // ----------------------------------------------------------------
        /// Schedule the passed thread (put it on the ready work queue)
        void schedule_work(threads::detail::thread_id_ref_type thrd, bool other_end)
        {
            ++work_items_count_.data_;
            ::pika::detail::tqmc_deb.debug(debug::detail::str<>("schedule_work"), "stealing",
                other_end, "D", debug::detail::dec<2>(holder_->domain_index_), "Q",
                debug::detail::dec<3>(queue_index_), "n",
                debug::detail::dec<4>(new_tasks_count_.data_), "w",
                debug::detail::dec<4>(work_items_count_.data_),
                debug::detail::threadinfo<threads::detail::thread_id_ref_type*>(&thrd));
            //
            work_items_.push(std::move(thrd), other_end);
#ifdef DEBUG_QUEUE_EXTRA
            debug_queue(work_items_);
#endif
        }

        ///////////////////////////////////////////////////////////////////////
        void on_start_thread(std::size_t /* num_thread */) {}
        void on_stop_thread(std::size_t /* num_thread */) {}
        void on_error(std::size_t /* num_thread */, std::exception_ptr const& /* e */) {}

        // pops all tasks off the queue, prints info and pushes them back on
        // just because we can't iterate over the queue/stack in general
#if defined(DEBUG_QUEUE_EXTRA)
        void debug_queue(work_items_type& q)
        {
            std::unique_lock<std::mutex> Lock(debug_mtx_);
            //
            work_items_type work_items_copy_;
            int x = 0;
            thread_description* thrd;
            ::pika::detail::tqmc_deb.debug(debug::detail::str<>("debug_queue"), "Pop work items");
            while (q.pop(thrd))
            {
                ::pika::detail::tqmc_deb.debug(debug::detail::str<>("debug_queue"), x++,
                    debug::detail::threadinfo<threads::detail::thread_data*>(thrd));
                work_items_copy_.push(thrd);
            }
            ::pika::detail::tqmc_deb.debug(debug::detail::str<>("debug_queue"), "Push work items");
            while (work_items_copy_.pop(thrd))
            {
                q.push(thrd);
                ::pika::detail::tqmc_deb.debug(debug::detail::str<>("debug_queue"), --x,
                    debug::detail::threadinfo<threads::detail::thread_data*>(thrd));
            }
            ::pika::detail::tqmc_deb.debug(debug::detail::str<>("debug_queue"), "Finished");
        }
#endif

    public:
        thread_queue_init_parameters parameters_;

        int const queue_index_;

        queue_holder_thread<thread_queue_type>* holder_;

        // count of new tasks to run, separate to new cache line to avoid false
        // sharing

        task_items_type new_task_items_;
        work_items_type work_items_;

        pika::concurrency::detail::cache_line_data<std::atomic<std::int32_t>> new_tasks_count_;
        pika::concurrency::detail::cache_line_data<std::atomic<std::int32_t>> work_items_count_;

#ifdef DEBUG_QUEUE_EXTRA
        std::mutex debug_mtx_;
#endif
    };

}    // namespace pika::threads::detail
