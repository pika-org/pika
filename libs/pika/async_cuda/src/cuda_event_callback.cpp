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
#include <pika/async_cuda/cuda_event.hpp>
#include <pika/async_cuda/detail/cuda_debug.hpp>
#include <pika/async_cuda/detail/cuda_event_callback.hpp>
#include <pika/concurrency/concurrentqueue.hpp>
#include <pika/synchronization/spinlock.hpp>
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

    class cuda_event_queue
    {
    public:
        using event_callback_queue_type =
            concurrency::detail::ConcurrentQueue<event_callback>;
        using mutex_type = pika::lcos::local::spinlock;
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

            // Don't poll if another thread is already polling
            std::unique_lock<mutex_type> lk(vector_mtx, std::try_to_lock);
            if (!lk.owns_lock())
            {
                if (cud_debug.is_enabled())
                {
                    static auto poll_deb = cud_debug.make_timer(
                        1, debug::detail::str<>("Poll - lock failed"));
                    cud_debug.timed(poll_deb, "enqueued events",
                        debug::detail::dec<3>(get_number_of_enqueued_events()),
                        "active events",
                        debug::detail::dec<3>(get_number_of_active_events()));
                }
                return polling_status::idle;
            }

            if (cud_debug.is_enabled())
            {
                static auto poll_deb = cud_debug.make_timer(
                    1, debug::detail::str<>("Poll - lock success"));
                cud_debug.timed(poll_deb, "enqueued events",
                    debug::detail::dec<3>(get_number_of_enqueued_events()),
                    "active events",
                    debug::detail::dec<3>(get_number_of_active_events()));
            }

            // Grab the handle to the event pool so we can return completed events
            cuda_event_pool& pool = cuda_event_pool::get_event_pool();

            // Iterate over our list of events and see if any have completed
            event_callback_vector.erase(
                std::remove_if(event_callback_vector.begin(),
                    event_callback_vector.end(),
                    [&](event_callback& continuation) {
                        whip::error_t status = whip::success;

                        try
                        {
                            bool ready = whip::event_ready(continuation.event);

                            // If the event is not yet ready, do nothing
                            if (!ready)
                            {
                                return false;
                            }
                        }
                        catch (whip::exception const& e)
                        {
                            status = e.get_error();
                        }

                        // Forward successes and other errors to the callback
                        cud_debug.debug(
                            debug::detail::str<>("set ready vector"), "event",
                            debug::detail::hex<8>(continuation.event),
                            "enqueued events",
                            debug::detail::dec<3>(
                                get_number_of_enqueued_events()),
                            "active events",
                            debug::detail::dec<3>(
                                get_number_of_active_events()));
                        continuation.f(status);
                        pool.push(PIKA_MOVE(continuation.event));
                        return true;
                    }),
                event_callback_vector.end());
            active_events_counter = event_callback_vector.size();

            detail::event_callback continuation;
            while (event_callback_queue.try_dequeue(continuation))
            {
                whip::error_t status = whip::success;

                try
                {
                    bool ready = whip::event_ready(continuation.event);

                    if (!ready)
                    {
                        add_to_event_callback_vector(PIKA_MOVE(continuation));
                        continue;
                    }
                }
                catch (whip::exception const& e)
                {
                    status = e.get_error();
                }

                cud_debug.debug(debug::detail::str<>("set ready queue"),
                    "event", debug::detail::hex<8>(continuation.event),
                    "enqueued events",
                    debug::detail::dec<3>(get_number_of_enqueued_events()),
                    "active events",
                    debug::detail::dec<3>(get_number_of_active_events()));
                continuation.f(status);
                pool.push(PIKA_MOVE(continuation.event));
            }

            using pika::threads::detail::polling_status;
            return event_callback_vector.empty() ? polling_status::idle :
                                                   polling_status::busy;
        }

        void add_to_event_callback_queue(
            event_callback_function_type&& f, whip::stream_t stream)
        {
            whip::event_t event;
            if (!cuda_event_pool::get_event_pool().pop(event))
            {
                PIKA_THROW_EXCEPTION(pika::error::invalid_status,
                    "add_to_event_callback_queue", "could not get an event");
            }
            whip::event_record(event, stream);

            event_callback continuation{event, PIKA_MOVE(f)};

            PIKA_ASSERT_MSG(get_register_polling_count() != 0,
                "CUDA event polling has not been enabled on any pool. Make "
                "sure that CUDA event polling is enabled on at least one "
                "thread pool.");

            event_callback_queue.enqueue(PIKA_MOVE(continuation));

            cud_debug.debug(debug::detail::str<>("event queued"), "event",
                debug::detail::hex<8>(continuation.event), "enqueued events",
                debug::detail::dec<3>(get_number_of_enqueued_events()),
                "active events",
                debug::detail::dec<3>(get_number_of_active_events()));
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
            return active_events_counter;
        }

        void add_to_event_callback_vector(event_callback&& continuation)
        {
            event_callback_vector.push_back(PIKA_MOVE(continuation));
            ++active_events_counter;

            cud_debug.debug(debug::detail::str<>(
                                "event callback moved from queue to vector"),
                "event", debug::detail::hex<8>(continuation.event),
                "enqueued events",
                debug::detail::dec<3>(get_number_of_enqueued_events()),
                "active events",
                debug::detail::dec<3>(get_number_of_active_events()));
        }

        event_callback_queue_type event_callback_queue;
        mutex_type vector_mtx;
        std::atomic<std::size_t> active_events_counter{0};
        event_callback_vector_type event_callback_vector;
    };

    class cuda_event_queue_holder
    {
    public:
        pika::threads::detail::polling_status poll()
        {
            auto hp_status = hp_queue.poll();
            auto np_status = np_queue.poll();

            using pika::threads::detail::polling_status;

            return np_status == polling_status::busy ||
                    hp_status == polling_status::busy ?
                polling_status::busy :
                polling_status::idle;
        }

        void add_to_event_callback_queue(event_callback_function_type&& f,
            whip::stream_t stream, pika::execution::thread_priority priority)
        {
            auto* queue = &np_queue;
            if (priority >= pika::execution::thread_priority::high)
            {
                queue = &hp_queue;
            }

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

    void add_event_callback(event_callback_function_type&& f,
        whip::stream_t stream, pika::execution::thread_priority priority)
    {
        get_cuda_event_queue_holder().add_to_event_callback_queue(
            std::move(f), stream, priority);
    }

    void add_event_callback(
        event_callback_function_type&& f, cuda_stream const& stream)
    {
        add_event_callback(std::move(f), stream.get(), stream.get_priority());
    }

    pika::threads::detail::polling_status poll()
    {
        return get_cuda_event_queue_holder().poll();
    }

    std::size_t get_work_count()
    {
        return get_cuda_event_queue_holder().get_work_count();
    }

    // -------------------------------------------------------------
    void register_polling(pika::threads::detail::thread_pool_base& pool)
    {
#if defined(PIKA_DEBUG)
        ++get_register_polling_count();
#endif
        cud_debug.debug(debug::detail::str<>("enable polling"));
        auto* sched = pool.get_scheduler();
        sched->set_cuda_polling_functions(
            &pika::cuda::experimental::detail::poll, &get_work_count);
    }

    // -------------------------------------------------------------
    void unregister_polling(pika::threads::detail::thread_pool_base& pool)
    {
#if defined(PIKA_DEBUG)
        {
            PIKA_ASSERT_MSG(get_work_count() == 0,
                "CUDA event polling was disabled while there are unprocessed "
                "CUDA events. Make sure CUDA event polling is not disabled too "
                "early.");
        }
#endif
        cud_debug.debug(debug::detail::str<>("disable polling"));
        auto* sched = pool.get_scheduler();
        sched->clear_cuda_polling_function();
    }
}    // namespace pika::cuda::experimental::detail
