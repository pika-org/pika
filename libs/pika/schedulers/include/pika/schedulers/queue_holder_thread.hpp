//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/debugging/print.hpp>
#include <pika/schedulers/lockfree_queue_backends.hpp>
#include <pika/threading_base/print.hpp>
#include <pika/threading_base/scheduler_base.hpp>
#include <pika/threading_base/thread_data.hpp>
#include <pika/threading_base/thread_data_stackful.hpp>
#include <pika/threading_base/thread_data_stackless.hpp>
#include <pika/threading_base/thread_queue_init_parameters.hpp>
#include <pika/type_support/unused.hpp>

#include <fmt/format.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iomanip>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#if !defined(QUEUE_HOLDER_THREAD_DEBUG)
# if defined(PIKA_DEBUG)
#  define QUEUE_HOLDER_THREAD_DEBUG false
# else
#  define QUEUE_HOLDER_THREAD_DEBUG false
# endif
#endif

namespace pika::detail {
    static pika::debug::detail::enable_print<QUEUE_HOLDER_THREAD_DEBUG> tq_deb("QH_THRD");
}

// ------------------------------------------------------------
namespace pika::threads::detail {

    // apply the modulo operator only when needed
    // (i.e. when the input is greater than the ceiling)
    // NB: the numbers must be positive
    PIKA_FORCEINLINE std::size_t fast_mod(std::size_t const input, std::size_t const ceil)
    {
        return input >= ceil ? input % ceil : input;
    }

    enum : std::size_t
    {
        max_thread_count = 1000
    };
    enum : std::size_t
    {
        round_robin_rollover = 1
    };

    // ----------------------------------------------------------------
    // Helper class to hold a set of queues.
    // ----------------------------------------------------------------
    template <typename QueueType>
    struct queue_holder_thread
    {
        using thread_holder_type = queue_holder_thread<QueueType>;

        // Queues that will store actual work for this thread
        // They might be shared between cores, so we use a pointer to
        // reference them
        QueueType* const bp_queue_;
        QueueType* const hp_queue_;
        QueueType* const np_queue_;
        QueueType* const lp_queue_;

        // these are the domain and local thread queue ids for the container
        std::size_t const domain_index_;
        std::size_t const queue_index_;
        std::size_t const thread_num_;
        // a mask that hold a bit per queue to indicate ownership of the queue
        std::size_t const owner_mask_;

        // we must use OS mutexes here because we cannot suspend a pika
        // thread whilst processing the Queues for that thread, this code
        // is running at the OS level in effect.
        using mutex_type = std::mutex;
        using scoped_lock = std::unique_lock<mutex_type>;

        // mutex protecting the thread map
        mutable pika::concurrency::detail::cache_line_data<mutex_type> thread_map_mtx_;

        // every thread maintains lists of free thread data objects
        // sorted by their stack sizes
        using thread_heap_type = std::list<threads::detail::thread_id_type,
            pika::detail::internal_allocator<threads::detail::thread_id_type>>;

        thread_heap_type thread_heap_small_;
        thread_heap_type thread_heap_medium_;
        thread_heap_type thread_heap_large_;
        thread_heap_type thread_heap_huge_;
        thread_heap_type thread_heap_nostack_;

        // these ought to be atomic, but if we get a race and assign a thread
        // to queue N instead of N+1 it doesn't really matter

        mutable pika::concurrency::detail::cache_line_data<std::tuple<std::size_t, std::size_t>>
            rollover_counters_;

        // ----------------------------------------------------------------
        // ----------------------------------------------------------------
        // ----------------------------------------------------------------

        static pika::detail::internal_allocator<threads::detail::thread_data> thread_alloc_;

        using task_description = threads::detail::thread_init_data;

        // -------------------------------------
        // thread map stores every task in this queue set
        // this is the type of a map holding all threads (except terminated)
        using thread_map_type = std::unordered_set<threads::detail::thread_id_type,
            std::hash<threads::detail::thread_id_type>,
            std::equal_to<threads::detail::thread_id_type>,
            pika::detail::internal_allocator<threads::detail::thread_id_type>>;
        thread_map_type thread_map_;

        mutable pika::concurrency::detail::cache_line_data<std::atomic<std::int32_t>>
            thread_map_count_;

#ifdef PIKA_HAVE_THREAD_STACK_MMAP
        // -------------------------------------
        // terminated tasks
        // completed tasks that can be reused (stack space etc)
        using terminated_items_type = lockfree_fifo::apply<threads::detail::thread_data*>::type;
        terminated_items_type terminated_items_;
        mutable pika::concurrency::detail::cache_line_data<std::atomic<std::int32_t>>
            terminated_items_count_;
#endif

        thread_queue_init_parameters parameters_;
        std::thread::id owner_id_;

        // ------------------------------------------------------------
        struct queue_mc_print
        {
            QueueType const* const q_;
            explicit queue_mc_print(QueueType const* const q)
              : q_(q)
            {
            }
            //
            friend std::ostream& operator<<(std::ostream& os, queue_mc_print const& d)
            {
                os << "n " << debug::detail::dec<3>(d.q_->new_tasks_count_.data_) << " w "
                   << debug::detail::dec<3>(d.q_->work_items_count_.data_);
                return os;
            }
        };

        struct queue_data_print
        {
            queue_holder_thread const* q_;
            explicit queue_data_print(queue_holder_thread const* q)
              : q_(q)
            {
            }
            //
            friend std::ostream& operator<<(std::ostream& os, queue_data_print const& d)
            {
                os << "D " << debug::detail::dec<2>(d.q_->domain_index_) << " Q "
                   << debug::detail::dec<3>(d.q_->queue_index_) << " TM "
                   << debug::detail::dec<3>(d.q_->thread_map_count_.data_) << " [BP "
                   << queue_mc_print(d.q_->bp_queue_) << "] [HP " << queue_mc_print(d.q_->hp_queue_)
                   << "] [NP " << queue_mc_print(d.q_->np_queue_) << "] [LP "
                   << queue_mc_print(d.q_->lp_queue_) << "]"
#ifdef PIKA_HAVE_THREAD_STACK_MMAP
                   << " T " << debug::detail::dec<3>(d.q_->terminated_items_count_.data_)
#endif
                    ;
                return os;
            }
        };
        // ------------------------------------------------------------

        // ----------------------------------------------------------------
        // ----------------------------------------------------------------
        // ----------------------------------------------------------------
        // NOLINTBEGIN(bugprone-easily-swappable-parameters)
        queue_holder_thread(QueueType* bp_queue, QueueType* hp_queue, QueueType* np_queue,
            QueueType* lp_queue, std::size_t domain, std::size_t queue, std::size_t thread_num,
            std::size_t owner, thread_queue_init_parameters const& init, std::thread::id owner_id)
          // NOLINTEND(bugprone-easily-swappable-parameters)
          : bp_queue_(bp_queue)
          , hp_queue_(hp_queue)
          , np_queue_(np_queue)
          , lp_queue_(lp_queue)
          , domain_index_(domain)
          , queue_index_(queue)
          , thread_num_(thread_num)
          , owner_mask_(owner)
#ifdef PIKA_HAVE_THREAD_STACK_MMAP
          , terminated_items_(max_thread_count)
#endif
          , parameters_(init)
          , owner_id_(owner_id)
        {
            rollover_counters_.data_ = std::make_tuple(queue_index_, round_robin_rollover);
            ::pika::detail::tq_deb.debug(debug::detail::str<>("construct"), "D",
                debug::detail::dec<2>(domain_index_), "Q", debug::detail::dec<3>(queue_index_),
                "Rollover counter", debug::detail::dec<>(std::get<0>(rollover_counters_.data_)),
                debug::detail::dec<>(std::get<1>(rollover_counters_.data_)));
            thread_map_count_.data_ = 0;
#ifdef PIKA_HAVE_THREAD_STACK_MMAP
            terminated_items_count_.data_ = 0;
#endif
            if (bp_queue_) bp_queue_->set_holder(this);
            if (hp_queue_) hp_queue_->set_holder(this);
            if (np_queue_) np_queue_->set_holder(this);
            if (lp_queue_) lp_queue_->set_holder(this);
        }

        // ----------------------------------------------------------------
        ~queue_holder_thread()
        {
            if (owns_bp_queue()) delete bp_queue_;
            if (owns_hp_queue()) delete hp_queue_;
            if (owns_np_queue()) delete np_queue_;
            if (owns_lp_queue()) delete lp_queue_;
            //
            for (auto t : thread_heap_small_) deallocate(threads::detail::get_thread_id_data(t));

            for (auto t : thread_heap_medium_) deallocate(threads::detail::get_thread_id_data(t));

            for (auto t : thread_heap_large_) deallocate(threads::detail::get_thread_id_data(t));

            for (auto t : thread_heap_huge_) deallocate(threads::detail::get_thread_id_data(t));

            for (auto t : thread_heap_nostack_) deallocate(threads::detail::get_thread_id_data(t));
        }

        // ----------------------------------------------------------------
        inline bool owns_bp_queue() const { return bp_queue_ && ((owner_mask_ & 1) != 0); }

        // ----------------------------------------------------------------
        inline bool owns_hp_queue() const { return hp_queue_ && ((owner_mask_ & 2) != 0); }

        // ----------------------------------------------------------------
        inline bool owns_np_queue() const { return ((owner_mask_ & 4) != 0); }

        // ----------------------------------------------------------------
        inline bool owns_lp_queue() const { return lp_queue_ && ((owner_mask_ & 8) != 0); }

        // ------------------------------------------------------------
        // return the next round robin thread index across all workers
        // using a batching of N per worker before incrementing
        inline std::size_t worker_next(std::size_t const workers) const
        {
            ::pika::detail::tq_deb.debug(debug::detail::str<>("worker_next"), "Rollover counter ",
                debug::detail::dec<4>(std::get<0>(rollover_counters_.data_)),
                debug::detail::dec<4>(std::get<1>(rollover_counters_.data_)), "workers",
                debug::detail::dec<4>(workers));
            if (--std::get<1>(rollover_counters_.data_) == 0)
            {
                std::get<1>(rollover_counters_.data_) = round_robin_rollover;
                std::get<0>(rollover_counters_.data_) =
                    fast_mod(std::get<0>(rollover_counters_.data_) + 1, workers);
            }
            return std::get<0>(rollover_counters_.data_);
        }

        // ------------------------------------------------------------
        void schedule_thread(threads::detail::thread_id_ref_type thrd,
            execution::thread_priority priority, bool other_end = false)
        {
            if (bp_queue_ && (priority == execution::thread_priority::bound))
            {
                ::pika::detail::tq_deb.debug(debug::detail::str<>("schedule_thread"),
                    queue_data_print(this),
                    debug::detail::threadinfo<threads::detail::thread_id_ref_type*>(&thrd),
                    "queueing execution::thread_priority::bound");
                bp_queue_->schedule_work(std::move(thrd), other_end);
            }
            else if (hp_queue_ &&
                (priority == execution::thread_priority::high ||
                    priority == execution::thread_priority::high_recursive ||
                    priority == execution::thread_priority::boost))
            {
                ::pika::detail::tq_deb.debug(debug::detail::str<>("schedule_thread"),
                    queue_data_print(this),
                    debug::detail::threadinfo<threads::detail::thread_id_ref_type*>(&thrd),
                    "queueing execution::thread_priority::high");
                hp_queue_->schedule_work(std::move(thrd), other_end);
            }
            else if (lp_queue_ && (priority == execution::thread_priority::low))
            {
                ::pika::detail::tq_deb.debug(debug::detail::str<>("schedule_thread"),
                    queue_data_print(this),
                    debug::detail::threadinfo<threads::detail::thread_id_ref_type*>(&thrd),
                    "queueing execution::thread_priority::low");
                lp_queue_->schedule_work(std::move(thrd), other_end);
            }
            else
            {
                ::pika::detail::tq_deb.debug(debug::detail::str<>("schedule_thread"),
                    queue_data_print(this),
                    debug::detail::threadinfo<threads::detail::thread_id_ref_type*>(&thrd),
                    "queueing execution::thread_priority::normal");
                np_queue_->schedule_work(std::move(thrd), other_end);
            }
        }

        // ----------------------------------------------------------------
#ifdef PIKA_HAVE_THREAD_STACK_MMAP
        bool cleanup_terminated(std::size_t thread_num, bool delete_all)
        {
            // clang-format off
            if (thread_num!=thread_num_) {
                ::pika::detail::tq_deb.error(debug::detail::str<>("assertion fail")
                             , "thread_num", thread_num
                             , "thread_num_", thread_num_
                             , "queue_index_", queue_index_
                             , queue_data_print(this)
                             );
            }
            // clang-format on
            PIKA_ASSERT(thread_num == thread_num_);

            if (terminated_items_count_.data_.load(std::memory_order_relaxed) == 0) return true;

            scoped_lock lk(thread_map_mtx_.data_);

            if (delete_all)
            {
                // delete all threads
                threads::detail::thread_data* todelete;
                while (terminated_items_.pop(todelete))
                {
                    --terminated_items_count_.data_;
                    ::pika::detail::tq_deb.debug(debug::detail::str<>("cleanup"), "delete",
                        queue_data_print(this),
                        debug::detail::threadinfo<threads::detail::thread_data*>(todelete));
                    threads::detail::thread_id_type tid(todelete);
                    remove_from_thread_map(tid, true);
                }
            }
            else
            {
                // delete only this many threads
                std::int64_t delete_count = static_cast<std::int64_t>(
                    terminated_items_count_.data_.load(std::memory_order_relaxed) / 2);

                ::pika::detail::tq_deb.debug(debug::detail::str<>("cleanup"), "recycle",
                    "delete_count", debug::detail::dec<3>(delete_count));

                threads::detail::thread_data* todelete;
                while (delete_count && terminated_items_.pop(todelete))
                {
                    threads::detail::thread_id_type tid(todelete);
                    --terminated_items_count_.data_;
                    remove_from_thread_map(tid, false);
                    ::pika::detail::tq_deb.debug(debug::detail::str<>("cleanup"), "recycle",
                        queue_data_print(this),
                        debug::detail::threadinfo<threads::detail::thread_id_type*>(&tid));
                    recycle_thread(tid);
                    --delete_count;
                }
            }
            return terminated_items_count_.data_.load(std::memory_order_relaxed) == 0;
        }
#endif

        // ----------------------------------------------------------------
        void create_thread(threads::detail::thread_init_data& data,
            threads::detail::thread_id_ref_type* tid, std::size_t thread_num, error_code& ec)
        {
            PIKA_ASSERT((!data.run_now || (data.run_now && thread_num == thread_num_)));
            if (data.run_now && thread_num != thread_num_) { data.run_now = false; }

            // create the thread using priority to select queue
            if (data.priority == execution::thread_priority::normal)
            {
                ::pika::detail::tq_deb.debug(debug::detail::str<>("create_thread "),
                    queue_data_print(this), "execution::thread_priority::normal", "run_now ",
                    data.run_now);
                return np_queue_->create_thread(data, tid, ec);
            }
            else if (bp_queue_ && (data.priority == execution::thread_priority::bound))
            {
                ::pika::detail::tq_deb.debug(debug::detail::str<>("create_thread "),
                    queue_data_print(this), "execution::thread_priority::bound", "run_now ",
                    data.run_now);
                return bp_queue_->create_thread(data, tid, ec);
            }
            else if (hp_queue_ &&
                (data.priority == execution::thread_priority::high ||
                    data.priority == execution::thread_priority::high_recursive ||
                    data.priority == execution::thread_priority::boost))
            {
                // boosted threads return to normal after being queued
                if (data.priority == execution::thread_priority::boost)
                {
                    data.priority = execution::thread_priority::normal;
                }
                ::pika::detail::tq_deb.debug(debug::detail::str<>("create_thread "),
                    queue_data_print(this), "execution::thread_priority::high", "run_now ",
                    data.run_now);
                return hp_queue_->create_thread(data, tid, ec);
            }
            else if (lp_queue_ && (data.priority == execution::thread_priority::low))
            {
                ::pika::detail::tq_deb.debug(debug::detail::str<>("create_thread "),
                    queue_data_print(this), "execution::thread_priority::low", "run_now ",
                    data.run_now);
                return lp_queue_->create_thread(data, tid, ec);
            }

            ::pika::detail::tq_deb.error(debug::detail::str<>("create_thread "), "priority?");
            std::terminate();
        }

        // ----------------------------------------------------------------
        // Not thread safe. This function must only be called by the
        // thread that owns the holder object.
        // Creates a thread_data object using information from
        // thread_init_data .
        // If a thread data object is available on one of the heaps
        // it will use that, otherwise a new one is created.
        // Heaps store data ordered/sorted by stack size
        void create_thread_object(
            threads::detail::thread_id_ref_type& tid, threads::detail::thread_init_data& data)
        {
            PIKA_ASSERT(data.stacksize >= execution::thread_stacksize::minimal);
            PIKA_ASSERT(data.stacksize <= execution::thread_stacksize::nostack);

            std::ptrdiff_t const stacksize = data.scheduler_base->get_stack_size(data.stacksize);

            thread_heap_type* heap = nullptr;
            if (stacksize == parameters_.small_stacksize_) { heap = &thread_heap_small_; }
            else if (stacksize == parameters_.medium_stacksize_) { heap = &thread_heap_medium_; }
            else if (stacksize == parameters_.large_stacksize_) { heap = &thread_heap_large_; }
            else if (stacksize == parameters_.huge_stacksize_) { heap = &thread_heap_huge_; }
            else if (stacksize == parameters_.nostack_stacksize_) { heap = &thread_heap_nostack_; }
            PIKA_ASSERT(heap);

            if (data.initial_state ==
                    threads::detail::thread_schedule_state::pending_do_not_schedule ||
                data.initial_state == threads::detail::thread_schedule_state::pending_boost)
            {
                data.initial_state = threads::detail::thread_schedule_state::pending;
            }

            // ASAN gets confused by reusing threads/stacks
#if !defined(PIKA_HAVE_ADDRESS_SANITIZER)

            // Check for an unused thread object.
            if (!heap->empty())
            {
                // Take ownership of the thread object and rebind it.
                tid = heap->front();
                heap->pop_front();
                threads::detail::get_thread_id_data(tid)->rebind(data);
                ::pika::detail::tq_deb.debug(debug::detail::str<>("create_thread_object"), "rebind",
                    queue_data_print(this),
                    debug::detail::threadinfo<threads::detail::thread_id_ref_type*>(&tid));
            }
            else
#else
            PIKA_UNUSED(heap);
#endif
            {
                // Allocate a new thread object.
                threads::detail::thread_data* p = nullptr;
                if (stacksize == parameters_.nostack_stacksize_)
                {
                    p = threads::detail::thread_data_stackless::create(data, this, stacksize);
                }
                else { p = threads::detail::thread_data_stackful::create(data, this, stacksize); }
                tid = threads::detail::thread_id_ref_type(p, threads::detail::thread_id_addref::no);

                ::pika::detail::tq_deb.debug(debug::detail::str<>("create_thread_object"), "new",
                    queue_data_print(this),
                    debug::detail::threadinfo<threads::detail::thread_data*>(p));
            }
        }

#ifdef PIKA_HAVE_THREAD_STACK_MMAP
        // ----------------------------------------------------------------
        void recycle_thread(threads::detail::thread_id_type tid)
        {
            std::ptrdiff_t stacksize = threads::detail::get_thread_id_data(tid)->get_stack_size();

            if (stacksize == parameters_.small_stacksize_) { thread_heap_small_.push_front(tid); }
            else if (stacksize == parameters_.medium_stacksize_)
            {
                thread_heap_medium_.push_front(tid);
            }
            else if (stacksize == parameters_.large_stacksize_)
            {
                thread_heap_large_.push_front(tid);
            }
            else if (stacksize == parameters_.huge_stacksize_)
            {
                thread_heap_huge_.push_front(tid);
            }
            else if (stacksize == parameters_.nostack_stacksize_)
            {
                thread_heap_nostack_.push_front(tid);
            }
            else { PIKA_ASSERT_MSG(false, fmt::format("Invalid stack size {}", stacksize)); }
        }
#endif

        // ----------------------------------------------------------------
        static void deallocate(threads::detail::thread_data* p) { p->destroy(); }

        // ----------------------------------------------------------------
        void add_to_thread_map(threads::detail::thread_id_type tid)
        {
            scoped_lock lk(thread_map_mtx_.data_);

            // add a new entry in the map for this thread
            std::pair<thread_map_type::iterator, bool> p = thread_map_.insert(tid);

            if (/*PIKA_UNLIKELY*/ (!p.second))
            {
                std::string map_size = std::to_string(thread_map_.size());
                // threads::detail::thread_id_type tid2 = *(p.first);
                // threads::detail::thread_data* td =
                //     threads::detail::get_thread_id_data(tid2);
                // std::string prev = fmt::format("{}", td);

                ::pika::detail::tq_deb.error(debug::detail::str<>("map add"),
                    "Couldn't add new thread to the thread map", queue_data_print(this),
                    debug::detail::threadinfo<threads::detail::thread_id_type*>(&tid));

                lk.unlock();
                PIKA_THROW_EXCEPTION(pika::error::out_of_memory,
                    "queue_holder_thread::add_to_thread_map",
                    "Couldn't add new thread to the thread map {}", map_size);
            }

            ++thread_map_count_.data_;

            ::pika::detail::tq_deb.debug(debug::detail::str<>("map add"), queue_data_print(this),
                debug::detail::threadinfo<threads::detail::thread_id_type*>(&tid));

            // this thread has to be in the map now
            PIKA_ASSERT(thread_map_.find(tid) != thread_map_.end());
        }

        // ----------------------------------------------------------------
        void remove_from_thread_map(threads::detail::thread_id_type tid, bool dealloc)
        {
            // this thread has to be in this map
            PIKA_ASSERT(thread_map_.find(tid) != thread_map_.end());

            PIKA_ASSERT(thread_map_count_.data_ >= 0);

            bool deleted = thread_map_.erase(tid) != 0;
            PIKA_ASSERT(deleted);
            (void) deleted;

            ::pika::detail::tq_deb.debug(debug::detail::str<>("map remove"), queue_data_print(this),
                debug::detail::threadinfo<threads::detail::thread_id_type*>(&tid));

            if (dealloc) { deallocate(threads::detail::get_thread_id_data(tid)); }
            --thread_map_count_.data_;
        }

        // ----------------------------------------------------------------
        bool get_next_thread_HP(
            threads::detail::thread_id_ref_type& thrd, bool stealing, bool check_new) PIKA_HOT
        {
            // only take from BP queue if we are not stealing
            if (!stealing && bp_queue_ && bp_queue_->get_next_thread(thrd, stealing, check_new))
            {
                ::pika::detail::tq_deb.debug(debug::detail::str<>("next_thread_BP"),
                    queue_data_print(this),
                    debug::detail::threadinfo<threads::detail::thread_id_ref_type*>(&thrd),
                    "execution::thread_priority::bound");
                return true;
            }

            if (hp_queue_ && hp_queue_->get_next_thread(thrd, stealing, check_new))
            {
                ::pika::detail::tq_deb.debug(debug::detail::str<>("get_next_thread_HP"),
                    queue_data_print(this),
                    debug::detail::threadinfo<threads::detail::thread_id_ref_type*>(&thrd),
                    "execution::thread_priority::high");
                return true;
            }
            // if we're out of work in the main queues,
            debug_queues("get_next_thread");
            return false;
        }

        // ----------------------------------------------------------------
        bool get_next_thread(threads::detail::thread_id_ref_type& thrd, bool stealing) PIKA_HOT
        {
            if (np_queue_->get_next_thread(thrd, stealing))
            {
                ::pika::detail::tq_deb.debug(debug::detail::str<>("next_thread_NP"),
                    queue_data_print(this),
                    debug::detail::threadinfo<threads::detail::thread_id_ref_type*>(&thrd),
                    "execution::thread_priority::normal");
                return true;
            }

            if (lp_queue_ && lp_queue_->get_next_thread(thrd, stealing))
            {
                ::pika::detail::tq_deb.debug(debug::detail::str<>("next_thread_LP"),
                    queue_data_print(this),
                    debug::detail::threadinfo<threads::detail::thread_id_ref_type*>(&thrd),
                    "execution::thread_priority::low");
                return true;
            }
            // if we're out of work in the main queues,
            debug_queues("get_next_thread");
            return false;
        }

        // ----------------------------------------------------------------
        std::size_t add_new_HP(std::int64_t add_count, thread_holder_type* addfrom, bool stealing)
        {
            std::size_t added;
            if (owns_bp_queue() && !stealing)
            {
                added = bp_queue_->add_new(add_count, addfrom->bp_queue_, stealing);
                if (added > 0) return added;
            }

            if (owns_hp_queue())
            {
                added = hp_queue_->add_new(add_count, addfrom->hp_queue_, stealing);
                if (added > 0) return added;
            }
            return 0;
        }

        // ----------------------------------------------------------------
        std::size_t add_new(std::int64_t add_count, thread_holder_type* addfrom, bool stealing)
        {
            std::size_t added;
            if (owns_np_queue())
            {
                added = np_queue_->add_new(add_count, addfrom->np_queue_, stealing);
                if (added > 0) return added;
            }

            if (owns_lp_queue())
            {
                added = lp_queue_->add_new(add_count, addfrom->lp_queue_, stealing);
                if (added > 0) return added;
            }
            //
            static auto an_timed =
                ::pika::detail::tq_deb.make_timer(1, debug::detail::str<>("add_new"));
            ::pika::detail::tq_deb.timed(an_timed, "add", debug::detail::dec<3>(add_count),
                "owns bp, hp, np, lp", owns_bp_queue(), owns_hp_queue(), owns_np_queue(),
                owns_lp_queue(), "this", queue_data_print(this), "from", queue_data_print(addfrom));
            //
            return 0;
        }

        // ----------------------------------------------------------------
        inline std::size_t get_queue_length()
        {
            std::size_t count = 0;
            count += owns_bp_queue() ? bp_queue_->get_queue_length() : 0;
            count += owns_hp_queue() ? hp_queue_->get_queue_length() : 0;
            count += owns_np_queue() ? np_queue_->get_queue_length() : 0;
            count += owns_lp_queue() ? lp_queue_->get_queue_length() : 0;
            debug_queues("get_queue_length");
            return count;
        }

        // ----------------------------------------------------------------
        inline std::size_t get_thread_count_staged(execution::thread_priority priority) const
        {
            // Return thread count of one specific queue.
            switch (priority)
            {
            case execution::thread_priority::default_:
            {
                std::int64_t count = 0;
                count += owns_bp_queue() ? bp_queue_->get_queue_length_staged() : 0;
                count += owns_hp_queue() ? hp_queue_->get_queue_length_staged() : 0;
                count += owns_np_queue() ? np_queue_->get_queue_length_staged() : 0;
                count += owns_lp_queue() ? lp_queue_->get_queue_length_staged() : 0;
                return count;
            }
            case execution::thread_priority::bound:
            {
                return owns_bp_queue() ? bp_queue_->get_queue_length_staged() : 0;
            }
            case execution::thread_priority::low:
            {
                return owns_lp_queue() ? lp_queue_->get_queue_length_staged() : 0;
            }
            case execution::thread_priority::normal:
            {
                return owns_np_queue() ? np_queue_->get_queue_length_staged() : 0;
            }
            case execution::thread_priority::boost:
            case execution::thread_priority::high:
            case execution::thread_priority::high_recursive:
            {
                return owns_hp_queue() ? hp_queue_->get_queue_length_staged() : 0;
            }
            default:
            case execution::thread_priority::unknown:
            {
                PIKA_THROW_EXCEPTION(pika::error::bad_parameter,
                    "queue_holder_thread::get_thread_count_staged",
                    "unknown thread priority value (execution::thread_priority::unknown)");
            }
            }
            return 0;
        }

        // ----------------------------------------------------------------
        inline std::size_t get_thread_count_pending(execution::thread_priority priority) const
        {
            // Return thread count of one specific queue.
            switch (priority)
            {
            case execution::thread_priority::default_:
            {
                std::int64_t count = 0;
                count += owns_hp_queue() ? hp_queue_->get_queue_length_pending() : 0;
                count += owns_np_queue() ? np_queue_->get_queue_length_pending() : 0;
                count += owns_lp_queue() ? lp_queue_->get_queue_length_pending() : 0;
                return count;
            }
            case execution::thread_priority::bound:
            {
                return owns_bp_queue() ? bp_queue_->get_queue_length_pending() : 0;
            }
            case execution::thread_priority::low:
            {
                return owns_lp_queue() ? lp_queue_->get_queue_length_pending() : 0;
            }
            case execution::thread_priority::normal:
            {
                return owns_np_queue() ? np_queue_->get_queue_length_pending() : 0;
            }
            case execution::thread_priority::boost:
            case execution::thread_priority::high:
            case execution::thread_priority::high_recursive:
            {
                return owns_hp_queue() ? hp_queue_->get_queue_length_pending() : 0;
            }
            default:
            case execution::thread_priority::unknown:
            {
                PIKA_THROW_EXCEPTION(pika::error::bad_parameter,
                    "queue_holder_thread::get_thread_count_pending",
                    "unknown thread priority value (execution::thread_priority::unknown)");
            }
            }
            return 0;
        }

        // ----------------------------------------------------------------
        inline std::size_t get_thread_count(threads::detail::thread_schedule_state state =
                                                threads::detail::thread_schedule_state::unknown,
            execution::thread_priority priority = execution::thread_priority::default_) const
        {
            if (threads::detail::thread_schedule_state::terminated == state)
#ifdef PIKA_HAVE_THREAD_STACK_MMAP
                return terminated_items_count_.data_.load(std::memory_order_relaxed);
#else
                return 0;
#endif

            if (threads::detail::thread_schedule_state::staged == state)
                return get_thread_count_staged(priority);

            if (threads::detail::thread_schedule_state::pending == state)
                return get_thread_count_pending(priority);

            if (threads::detail::thread_schedule_state::unknown == state)
#ifdef PIKA_HAVE_THREAD_STACK_MMAP
                return thread_map_count_.data_.load(std::memory_order_relaxed) +
                    get_thread_count_staged(priority) -
                    terminated_items_count_.data_.load(std::memory_order_relaxed);
#else
                return thread_map_count_.data_.load(std::memory_order_relaxed) +
                    get_thread_count_staged(priority);
#endif

            // acquire lock only if absolutely necessary
            scoped_lock lk(thread_map_mtx_.data_);

            std::int64_t num_threads = 0;
            thread_map_type::const_iterator end = thread_map_.end();
            for (thread_map_type::const_iterator it = thread_map_.begin(); it != end; ++it)
            {
                if (threads::detail::get_thread_id_data(*it)->get_state().state() == state)
                    ++num_threads;
            }
            return num_threads;
        }

        // ------------------------------------------------------------
        /// Destroy the passed thread as it has been terminated
        void destroy_thread(
            threads::detail::thread_data* thrd, std::size_t thread_num, bool xthread)
        {
            // the thread must be destroyed by the same queue holder that created it
            PIKA_ASSERT(&thrd->get_queue<queue_holder_thread>() == this);
            //
#ifdef PIKA_HAVE_THREAD_STACK_MMAP
            ::pika::detail::tq_deb.debug(debug::detail::str<>("destroy"), "terminated_items push",
                "xthread", xthread, queue_data_print(this),
                debug::detail::threadinfo<threads::detail::thread_data*>(thrd));
            terminated_items_.push(thrd);
            std::int64_t count = ++terminated_items_count_.data_;

            if (!xthread && (count > parameters_.max_terminated_threads_))
            {
                cleanup_terminated(thread_num, false);    // clean up all terminated threads
            }
#else
            PIKA_UNUSED(thread_num);
            PIKA_UNUSED(xthread);

            threads::detail::thread_id_type tid(thrd);

            scoped_lock lk(thread_map_mtx_.data_);
            remove_from_thread_map(tid, true);
#endif
        }

        // ------------------------------------------------------------
        void abort_all_suspended_threads()
        {
            scoped_lock lk(thread_map_mtx_.data_);
            thread_map_type::iterator end = thread_map_.end();
            for (thread_map_type::iterator it = thread_map_.begin(); it != end; ++it)
            {
                if (threads::detail::get_thread_id_data(*it)->get_state().state() ==
                    threads::detail::thread_schedule_state::suspended)
                {
                    threads::detail::get_thread_id_data(*it)->set_state(
                        threads::detail::thread_schedule_state::pending,
                        pika::threads::detail::thread_restart_state::abort);
                    // np queue always exists so use that as priority doesn't matter
                    np_queue_->schedule_work(*it, true);
                }
            }
            throw std::runtime_error("This function needs to be reimplemented");
        }

        // ------------------------------------------------------------
        bool enumerate_threads(
            util::detail::function<bool(threads::detail::thread_id_type)> const& f,
            threads::detail::thread_schedule_state state =
                threads::detail::thread_schedule_state::unknown) const
        {
            std::uint64_t count = thread_map_count_.data_;
            if (state == threads::detail::thread_schedule_state::terminated)
            {
#ifdef PIKA_HAVE_THREAD_STACK_MMAP
                count = terminated_items_count_.data_;
#else
                count = 0;
#endif
            }
            else if (state == threads::detail::thread_schedule_state::staged)
            {
                PIKA_THROW_EXCEPTION(pika::error::bad_parameter,
                    "queue_holder_thread::iterate_threads",
                    "can't iterate over thread ids of staged threads");
                return false;
            }

            std::vector<threads::detail::thread_id_type> tids;
            tids.reserve(static_cast<std::size_t>(count));

            if (state == threads::detail::thread_schedule_state::unknown)
            {
                scoped_lock lk(thread_map_mtx_.data_);
                thread_map_type::const_iterator end = thread_map_.end();
                for (thread_map_type::const_iterator it = thread_map_.begin(); it != end; ++it)
                {
                    tids.push_back(*it);
                }
            }
            else
            {
                scoped_lock lk(thread_map_mtx_.data_);
                thread_map_type::const_iterator end = thread_map_.end();
                for (thread_map_type::const_iterator it = thread_map_.begin(); it != end; ++it)
                {
                    if (threads::detail::get_thread_id_data(*it)->get_state().state() == state)
                        tids.push_back(*it);
                }
            }

            // now invoke callback function for all matching threads
            for (threads::detail::thread_id_type const& id : tids)
            {
                if (!f(id)) return false;    // stop iteration
            }

            return true;
        }

        // ------------------------------------------------------------
        void debug_info()
        {
            ::pika::detail::tq_deb.debug(debug::detail::str<>("details"), "owner_mask_",
                debug::detail::bin<8>(owner_mask_), "D", debug::detail::dec<2>(domain_index_), "Q",
                debug::detail::dec<3>(queue_index_));
            ::pika::detail::tq_deb.debug(debug::detail::str<>("bp_queue"),
                debug::detail::hex<12, void*>(bp_queue_), "holder",
                debug::detail::hex<12, void*>(bp_queue_->holder_ ? bp_queue_->holder_ : nullptr));
            ::pika::detail::tq_deb.debug(debug::detail::str<>("hp_queue"),
                debug::detail::hex<12, void*>(hp_queue_), "holder",
                debug::detail::hex<12, void*>(hp_queue_->holder_ ? hp_queue_->holder_ : nullptr));
            ::pika::detail::tq_deb.debug(debug::detail::str<>("np_queue"),
                debug::detail::hex<12, void*>(np_queue_), "holder",
                debug::detail::hex<12, void*>(np_queue_->holder_ ? np_queue_->holder_ : nullptr));
            ::pika::detail::tq_deb.debug(debug::detail::str<>("lp_queue"),
                debug::detail::hex<12, void*>(lp_queue_), "holder",
                debug::detail::hex<12, void*>(lp_queue_->holder_ ? lp_queue_->holder_ : nullptr));
        }

        // ------------------------------------------------------------
        void debug_queues(char const* prefix)
        {
            static auto deb_queues =
                ::pika::detail::tq_deb.make_timer(1, debug::detail::str<>("debug_queues"));
            //
            ::pika::detail::tq_deb.timed(deb_queues, prefix, queue_data_print(this));
        }
    };

    template <typename QueueType>
    pika::detail::internal_allocator<threads::detail::thread_data>
        queue_holder_thread<QueueType>::thread_alloc_;
}    // namespace pika::threads::detail
