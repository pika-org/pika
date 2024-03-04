//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/async_cuda/detail/cuda_event_callback.hpp>
#include <pika/async_cuda_base/cuda_event.hpp>
#include <pika/async_cuda_base/detail/cuda_debug.hpp>
#include <pika/concurrency/concurrentqueue.hpp>
#include <pika/concurrency/spinlock.hpp>
#include <pika/datastructures/detail/small_vector.hpp>
#include <pika/resource_partitioner/detail/partitioner.hpp>
#include <pika/runtime/thread_pool_helpers.hpp>
#include <pika/threading_base/detail/global_activity_count.hpp>
#include <pika/threading_base/scheduler_base.hpp>
#include <pika/threading_base/thread_pool_base.hpp>

#include <whip.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace pika::cuda::experimental::detail {
#if defined(PIKA_DEBUG)
    std::atomic<std::size_t>& get_register_polling_count()
    {
        static std::atomic<std::size_t> register_polling_count{0};
        return register_polling_count;
    }
#endif

    // Holds a CUDA event and a callback. The callback is intended to be called when
    // the event is ready.
    struct event_callback
    {
        whip::event_t event;
        event_callback_function_type f;
    };

    // a struct we use temporarily to hold callbacks we can invoke
    struct ready_callback
    {
        whip::error_t status;
        event_callback_function_type f;
    };

    class cuda_event_queue
    {
    public:
        using event_callback_queue_type = concurrency::detail::ConcurrentQueue<event_callback>;
        using event_ready_queue_type = concurrency::detail::ConcurrentQueue<ready_callback>;
        using mutex_type = pika::concurrency::detail::spinlock;
        using event_callback_vector_type = std::vector<event_callback>;

        // Background progress function for async CUDA operations. Checks for
        // completed cudaEvent_t and calls the associated callback when ready.
        // We first process events that have been added to the vector of events,
        // which should be processed under a lock.  After that we process events
        // that have been added to the lockfree queue. If an event from the
        // queue is not ready it is added to the vector of events for later
        // checking.
        pika::threads::detail::polling_status poll()
        {
            using pika::threads::detail::polling_status;
            using namespace pika::debug::detail;

            // invoke ready callbacks without being under lock
            detail::ready_callback ready_callback_;
            while (ready_events.try_dequeue(ready_callback_))
            {
                ready_callback_.f(ready_callback_.status);
                pika::threads::detail::decrement_global_activity_count();
            }

            // locked section to access std::vector etc
            {
                // Don't poll if another thread is already polling
                std::unique_lock<mutex_type> lk(vector_mtx, std::try_to_lock);
                if (!lk.owns_lock())
                {
                    if (cud_debug<5>.is_enabled())
                    {
                        static auto poll_deb =
                            cud_debug<5>.make_timer(1, str<>("Poll - lock failed"));
                        cud_debug<5>.timed(poll_deb, "enqueued events",
                            dec<3>(get_number_of_enqueued_events()), "active events",
                            dec<3>(get_number_of_active_events()));
                    }
                    return polling_status::idle;
                }

                if (cud_debug<5>.is_enabled())
                {
                    static auto poll_deb = cud_debug<5>.make_timer(1, str<>("Poll - lock success"));
                    cud_debug<5>.timed(poll_deb, "enqueued events",
                        dec<3>(get_number_of_enqueued_events()), "active events",
                        dec<3>(get_number_of_active_events()));
                }

                // Grab the handle to the event pool so we can return completed events
                cuda_event_pool& pool = cuda_event_pool::get_event_pool();

                // Iterate over our list of events and see if any have completed
                event_callback_vector.erase(
                    std::remove_if(event_callback_vector.begin(), event_callback_vector.end(),
                        [&](event_callback& continuation) {
                            whip::error_t status = whip::success;
                            try
                            {
                                bool ready = whip::event_ready(continuation.event);
                                // If the event is not yet ready, do nothing
                                if (!ready)
                                {
                                    // do not be remove this item from the vector
                                    return false;
                                }
                            }
                            catch (whip::exception const& e)
                            {
                                status = e.get_error();
                            }

                            // Forward successes and other errors to the callback
                            PIKA_DETAIL_DP(cud_debug<5>,
                                debug(str<>("set ready vector"), "event",
                                    hex<8>(continuation.event), "enqueued events",
                                    dec<3>(get_number_of_enqueued_events()), "active events",
                                    dec<3>(get_number_of_active_events())));
                            // save callback to ready queue
                            ready_events.enqueue({status, PIKA_MOVE(continuation.f)});
                            // release the event handle
                            pool.push(PIKA_MOVE(continuation.event));
                            // this item can be removed from the vector
                            return true;
                        }),
                    event_callback_vector.end());
                active_events_counter = event_callback_vector.size();

                // now move unprocessed events to the vector if not ready
                detail::event_callback continuation;
                while (event_callback_queue.try_dequeue(continuation))
                {
                    whip::error_t status = whip::success;
                    try
                    {
                        if (!whip::event_ready(continuation.event))
                        {
                            add_to_event_callback_vector(PIKA_MOVE(continuation));
                            continue;
                        }
                    }
                    catch (whip::exception const& e)
                    {
                        status = e.get_error();
                    }

                    PIKA_DETAIL_DP(cud_debug<5>,
                        debug(debug::detail::str<>("set ready queue"), "event",
                            debug::detail::hex<8>(continuation.event), "enqueued events",
                            debug::detail::dec<3>(get_number_of_enqueued_events()), "active events",
                            debug::detail::dec<3>(get_number_of_active_events())));
                    // save callback to ready queue
                    ready_events.enqueue({status, PIKA_MOVE(continuation.f)});
                    // release the event handle
                    pool.push(PIKA_MOVE(continuation.event));
                }
            }    // end locked region

            // invoke any new ready callbacks without being under lock
            while (ready_events.try_dequeue(ready_callback_))
            {
                ready_callback_.f(ready_callback_.status);
                pika::threads::detail::decrement_global_activity_count();
            }

            using pika::threads::detail::polling_status;
            return get_number_of_active_events() == 0 ? polling_status::idle : polling_status::busy;
        }

        void add_to_event_callback_queue(event_callback_function_type&& f, whip::stream_t stream)
        {
            whip::event_t event;
            if (!cuda_event_pool::get_event_pool().pop(event))
            {
                PIKA_THROW_EXCEPTION(pika::error::invalid_status, "add_to_event_callback_queue",
                    "could not get an event");
            }
            PIKA_ASSERT(event != 0);
            whip::event_record(event, stream);

            PIKA_ASSERT_MSG(get_register_polling_count() != 0,
                "CUDA event polling has not been enabled on any pool. Make sure that CUDA event "
                "polling is enabled on at least one thread pool.");

            PIKA_DETAIL_DP(cud_debug<5>,
                debug(str<>("event queued"), "event", hex<8>(event), "enqueued events",
                    dec<3>(get_number_of_enqueued_events()), "active events",
                    dec<3>(get_number_of_active_events())));

            pika::threads::detail::increment_global_activity_count();

            event_callback_queue.enqueue({event, PIKA_MOVE(f)});
        }

        std::size_t get_work_count() const noexcept
        {
            std::size_t work_count = get_number_of_active_events();
            work_count += get_number_of_enqueued_events();

            return work_count;
        }

    private:
        std::size_t get_number_of_enqueued_events() const noexcept
        {
            return event_callback_queue.size_approx();
        }

        std::size_t get_number_of_active_events() const noexcept
        {
            return active_events_counter.load(std::memory_order_relaxed);
        }

        void add_to_event_callback_vector(event_callback&& continuation)
        {
            event_callback_vector.push_back(PIKA_MOVE(continuation));
            ++active_events_counter;

            PIKA_DETAIL_DP(cud_debug<5>,
                debug(str<>("event callback moved from queue to vector"), "event",
                    hex<8>(continuation.event), "enqueued events",
                    dec<3>(get_number_of_enqueued_events()), "active events",
                    dec<3>(get_number_of_active_events())));
        }

        event_callback_queue_type event_callback_queue;
        mutex_type vector_mtx;
        std::atomic<std::size_t> active_events_counter{0};
        event_callback_vector_type event_callback_vector;
        event_ready_queue_type ready_events;
    };

    class cuda_event_queue_holder
    {
    public:
        pika::threads::detail::polling_status poll()
        {
            auto hp_status = hp_queue.poll();
            auto np_status = np_queue.poll();

            using pika::threads::detail::polling_status;

            return np_status == polling_status::busy || hp_status == polling_status::busy ?
                polling_status::busy :
                polling_status::idle;
        }

        void add_to_event_callback_queue(event_callback_function_type&& f, whip::stream_t stream,
            pika::execution::thread_priority priority)
        {
            auto* queue = &np_queue;
            if (priority >= pika::execution::thread_priority::high) { queue = &hp_queue; }

            queue->add_to_event_callback_queue(std::move(f), stream);
        }

        std::size_t get_work_count() const noexcept
        {
            return hp_queue.get_work_count() + np_queue.get_work_count();
        }

    private:
        cuda_event_queue np_queue;
        cuda_event_queue hp_queue;
    };

    cuda_event_queue_holder& get_cuda_event_queue_holder()
    {
        static cuda_event_queue_holder holder;
        return holder;
    }

    void add_event_callback(event_callback_function_type&& f, whip::stream_t stream,
        pika::execution::thread_priority priority)
    {
        get_cuda_event_queue_holder().add_to_event_callback_queue(std::move(f), stream, priority);
    }

    void add_event_callback(event_callback_function_type&& f, cuda_stream const& stream)
    {
        add_event_callback(std::move(f), stream.get(), stream.get_priority());
    }

    pika::threads::detail::polling_status poll() { return get_cuda_event_queue_holder().poll(); }

    std::size_t get_work_count() { return get_cuda_event_queue_holder().get_work_count(); }

    // -------------------------------------------------------------
    void register_polling(pika::threads::detail::thread_pool_base& pool)
    {
#if defined(PIKA_DEBUG)
        ++get_register_polling_count();
#endif

        // pre-create CUDA/HIP events
        cuda_event_pool::get_event_pool().grow(cuda_event_pool::initial_events_in_pool);

        PIKA_DETAIL_DP(cud_debug<2>, debug(str<>("enable polling"), pool.get_pool_name()));
        auto* sched = pool.get_scheduler();
        sched->set_cuda_polling_functions(&pika::cuda::experimental::detail::poll, &get_work_count);
    }

    // -------------------------------------------------------------
    void unregister_polling(pika::threads::detail::thread_pool_base& pool)
    {
#if defined(PIKA_DEBUG)
        {
            PIKA_ASSERT_MSG(get_work_count() == 0,
                "CUDA event polling was disabled while there are unprocessed CUDA events. Make "
                "sure CUDA event polling is not disabled too early.");
        }
#endif
        PIKA_DETAIL_DP(cud_debug<2>, debug(str<>("disable polling"), pool.get_pool_name()));
        auto* sched = pool.get_scheduler();
        sched->clear_cuda_polling_function();
    }

    static std::string polling_pool_name = "default";

}    // namespace pika::cuda::experimental::detail

namespace pika::cuda::experimental {
    PIKA_EXPORT const std::string& get_pool_name()
    {
        if (pika::resource::pool_exists(detail::polling_pool_name))
        {
            return detail::polling_pool_name;
        }
        //
        return resource::get_partitioner().get_default_pool_name();
    }

    PIKA_EXPORT void set_pool_name(const std::string& name)
    {
        PIKA_DETAIL_DP(detail::cud_debug<2>, debug(debug::detail::str<>("set pool name"), name));
        detail::polling_pool_name = name;
    }
}    // namespace pika::cuda::experimental
