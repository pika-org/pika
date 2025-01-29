//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#if defined(PIKA_HAVE_STDEXEC)
# include <pika/execution_base/stdexec_forward.hpp>
#endif

#include <pika/assert.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/concurrency/detail/contiguous_index_queue.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/datastructures/variant.hpp>
#include <pika/execution/algorithms/bulk.hpp>
#include <pika/execution_base/completion_scheduler.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/executors/thread_pool_scheduler.hpp>
#include <pika/functional/bind_front.hpp>
#include <pika/functional/tag_invoke.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/iterator_support/traits/is_range.hpp>
#include <pika/threading_base/annotated_function.hpp>
#include <pika/threading_base/register_thread.hpp>
#include <pika/threading_base/thread_description.hpp>
#include <pika/threading_base/thread_num_tss.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika::thread_pool_bulk_detail {
    /// This sender represents bulk work that will be performed using the
    /// thread_pool_scheduler.
    ///
    /// The work is chunked into a number of chunks larger than the number
    /// of worker threads available on the underlying thread pool. The
    /// chunks are then assigned to worker thread-specific thread-safe index
    /// queues. One pika thread is spawned for each underlying worker (OS)
    /// thread. The pika thread is responsible for work in one queue. If the
    /// queue is empty, no pika thread will be spawned. Once the pika thread
    /// has finished working on its own queue, it will attempt to steal work
    /// from other queues. Since predecessor sender must complete on a pika
    /// thread (the completion scheduler is a thread_pool_scheduler;
    /// otherwise the customization defined in this file is not chosen) it
    /// will be reused as one of the worker threads.
    template <typename Sender, typename Shape, typename F>
    class thread_pool_bulk_sender
    {
    private:
        using shape_type = std::decay_t<Shape>;

        pika::execution::experimental::thread_pool_scheduler scheduler;
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
        PIKA_NO_UNIQUE_ADDRESS shape_type shape;
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;

    public:
        template <typename Sender_, typename Shape_, typename F_>
        thread_pool_bulk_sender(pika::execution::experimental::thread_pool_scheduler&& scheduler,
            Sender_&& sender, Shape_&& shape, F_&& f)
          : scheduler(std::move(scheduler))
          , sender(std::forward<Sender_>(sender))
          , shape(std::forward<Shape_>(shape))
          , f(std::forward<F_>(f))
        {
        }
        thread_pool_bulk_sender(thread_pool_bulk_sender&&) = default;
        thread_pool_bulk_sender(thread_pool_bulk_sender const&) = default;
        thread_pool_bulk_sender& operator=(thread_pool_bulk_sender&&) = default;
        thread_pool_bulk_sender& operator=(thread_pool_bulk_sender const&) = default;

        template <typename Tuple>
        struct value_types_helper
        {
            using type = pika::util::detail::transform_t<Tuple, std::decay>;
        };

        // PIKA_STDEXEC_SENDER_CONCEPT

#if defined(PIKA_HAVE_STDEXEC)
        PIKA_STDEXEC_SENDER_CONCEPT

        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types = pika::util::detail::transform_t<
            pika::execution::experimental::value_types_of_t<Sender,
                pika::execution::experimental::empty_env, Tuple, Variant>,
            value_types_helper>;

        template <template <typename...> class Variant>
        using error_types = pika::util::detail::unique_t<
            pika::util::detail::prepend_t<pika::execution::experimental::error_types_of_t<Sender,
                                              pika::execution::experimental::empty_env, Variant>,
                std::exception_ptr>>;

        using completion_signatures =
            pika::execution::experimental::transform_completion_signatures_of<Sender,
                pika::execution::experimental::empty_env,
                pika::execution::experimental::completion_signatures<
                    pika::execution::experimental::set_error_t(std::exception_ptr)>>;
#else
        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types =
            pika::util::detail::transform_t<typename pika::execution::experimental::sender_traits<
                                                Sender>::template value_types<Tuple, Variant>,
                value_types_helper>;

        template <template <typename...> class Variant>
        using error_types = pika::util::detail::unique_t<
            pika::util::detail::prepend_t<typename pika::execution::experimental::sender_traits<
                                              Sender>::template error_types<Variant>,
                std::exception_ptr>>;

        static constexpr bool sends_done = false;
#endif

    private:
        template <typename Receiver>
        struct operation_state
        {
            struct bulk_receiver
            {
                PIKA_STDEXEC_RECEIVER_CONCEPT

                operation_state* op_state;

                template <typename E>
                friend void tag_invoke(
                    pika::execution::experimental::set_error_t, bulk_receiver&& r, E&& e) noexcept
                {
                    pika::execution::experimental::set_error(
                        std::move(r.op_state->receiver), std::forward<E>(e));
                }

                friend void tag_invoke(
                    pika::execution::experimental::set_stopped_t, bulk_receiver&& r) noexcept
                {
                    pika::execution::experimental::set_stopped(std::move(r.op_state->receiver));
                };

                struct task_function;

                struct set_value_loop_visitor
                {
                    operation_state* const op_state;
                    task_function const* const task_f;

                    void operator()(pika::detail::monostate const&) const { PIKA_UNREACHABLE; }

                    // Perform the work in one chunk indexed by index.  The
                    // index represents a range of indices (iterators) in
                    // the given shape.
                    template <typename Ts>
                    void do_work_chunk(Ts& ts, std::uint32_t const index) const
                    {
                        auto const i_begin = static_cast<shape_type>(index) *
                            static_cast<shape_type>(task_f->chunk_size);
                        auto const i_end = (std::min)((static_cast<shape_type>(index) + 1) *
                                static_cast<shape_type>(task_f->chunk_size),
                            task_f->n);
                        for (auto i = i_begin; i < i_end; ++i)
                        {
                            std::apply(pika::util::detail::bind_front(op_state->f, i), ts);
                        }
                    }

                    // Visit the values sent from the predecessor sender.
                    // This function first tries to handle all chunks in the
                    // queue owned by worker_thread. It then tries to steal
                    // chunks from neighboring threads.
                    template <typename Ts,
                        typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<Ts>, pika::detail::monostate>>>
                    void operator()(Ts& ts) const
                    {
                        auto& local_queue = op_state->queues[task_f->worker_thread].data_;

                        // Handle local queue first
                        std::optional<std::uint32_t> index;
                        while ((index = local_queue.pop_left()))
                        {
                            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                            do_work_chunk(ts, *index);
                        }

                        // Then steal from neighboring queues
                        for (std::uint32_t offset = 1; offset < op_state->num_worker_threads;
                             ++offset)
                        {
                            std::size_t neighbor_worker_thread =
                                (task_f->worker_thread + offset) % op_state->num_worker_threads;
                            auto& neighbor_queue = op_state->queues[neighbor_worker_thread].data_;

                            while ((index = neighbor_queue.pop_right()))
                            {
                                // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                                do_work_chunk(ts, *index);
                            }
                        }
                    }
                };

                struct set_value_end_loop_visitor
                {
                    operation_state* const op_state;

                    void operator()(pika::detail::monostate&&) const { std::terminate(); }

                    // Visit the values sent from the predecessor sender.
                    // This function is called once all worker threads have
                    // processed their chunks and the connected receiver
                    // should be signalled.
                    template <typename Ts,
                        typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<Ts>, pika::detail::monostate>>>
                    void operator()(Ts&& ts) const
                    {
                        std::apply(
                            pika::util::detail::bind_front(pika::execution::experimental::set_value,
                                std::move(op_state->receiver)),
                            std::forward<Ts>(ts));
                    }
                };

                // This struct encapsulates the work done by one worker thread.
                struct task_function
                {
                    operation_state* const op_state;
                    std::decay_t<Shape> const n;
                    std::uint32_t const chunk_size;
                    std::uint32_t const worker_thread;

                    // Visit the values sent by the predecessor sender.
                    void do_work() const
                    {
                        pika::detail::visit(set_value_loop_visitor{op_state, this}, op_state->ts);
                    }

                    // Store an exception and mark that an exception was
                    // thrown in the operation state. This function assumes
                    // that there is a current exception.
                    void store_exception() const
                    {
                        if (!op_state->exception_thrown.exchange(true))
                        {
                            // NOLINTNEXTLINE(bugprone-throw-keyword-missing)
                            op_state->exception = std::current_exception();
                        }
                    }

                    // Finish the work for one worker thread. If this is not
                    // the last worker thread to finish, it will only
                    // decrement the counter. If it is the last thread it
                    // will call set_error if there is an exception.
                    // Otherwise it will call set_value on the connected
                    // receiver.
                    void finish() const
                    {
                        if (--(op_state->tasks_remaining) == 0)
                        {
                            if (op_state->exception_thrown)
                            {
                                PIKA_ASSERT(op_state->exception.has_value());
                                pika::execution::experimental::set_error(
                                    std::move(op_state->receiver),
                                    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                                    std::move(*(op_state->exception)));
                            }
                            else
                            {
                                pika::detail::visit(
                                    set_value_end_loop_visitor{op_state}, std::move(op_state->ts));
                            }
                        }
                    }

                    // Entry point for the worker thread. It will attempt to
                    // do its local work, catch any exceptions, and then
                    // call set_value or set_error on the connected
                    // receiver.
                    void operator()()
                    {
                        try
                        {
                            // If f has a different annotation than the
                            // current annotation it will be used for the
                            // duration of the work and reset once done.
                            // Otherwise the current annotation will be
                            // used.
                            pika::scoped_annotation ann(op_state->f);
                            do_work();
                        }
                        catch (...)
                        {
                            store_exception();
                        }

                        finish();
                    };
                };

                // Compute a chunk size given a number of worker threads and
                // a total number of items n. Returns a power-of-2 chunk
                // size that produces at most 8 and at least 4 chunks per
                // worker thread.
                static constexpr std::uint32_t get_chunk_size(
                    std::uint32_t const num_threads, std::decay_t<Shape> const n)
                {
                    std::uint32_t chunk_size = 1;
                    while (chunk_size * num_threads * 8 < static_cast<std::uint32_t>(n))
                    {
                        chunk_size *= 2;
                    }
                    return chunk_size;
                }

                // Initialize a queue for a worker thread.
                void init_queue(std::uint32_t const worker_thread, std::uint32_t const num_chunks)
                {
                    auto& queue = op_state->queues[worker_thread].data_;
                    auto const part_begin = static_cast<std::uint32_t>(
                        (worker_thread * num_chunks) / op_state->num_worker_threads);
                    auto const part_end = static_cast<std::uint32_t>(
                        ((worker_thread + 1) * num_chunks) / op_state->num_worker_threads);
                    queue.reset(part_begin, part_end);
                }

                // Spawn a task which will process a number of chunks. If
                // the queue contains no chunks no task will be spawned.
                void do_work_task(std::decay_t<Shape> const n, std::uint32_t const chunk_size,
                    std::uint32_t const worker_thread) const
                {
                    task_function task_f{this->op_state, n, chunk_size, worker_thread};

                    auto& queue = op_state->queues[worker_thread].data_;
                    if (queue.empty())
                    {
                        // If the queue is empty we don't spawn a task. We
                        // only signal that this "task" is ready.
                        task_f.finish();
                        return;
                    }

                    // Only apply hint if none was given.
                    auto hint = pika::execution::experimental::get_hint(op_state->scheduler);
                    if (hint == pika::execution::thread_schedule_hint())
                    {
                        hint = pika::execution::thread_schedule_hint(
                            pika::execution::thread_schedule_hint_mode::thread, worker_thread);
                    }

                    // Get the current thread description and apply it to
                    // the spawned task. Since this is a customization of
                    // bulk for thread_pool_scheduler we must currently be
                    // running on a pika thread.
                    PIKA_ASSERT(pika::threads::detail::get_self_id());
                    pika::detail::thread_description desc =
                        pika::threads::detail::get_thread_description(
                            pika::threads::detail::get_self_id());

                    // Spawn the task.
                    threads::detail::thread_init_data data(
                        threads::detail::make_thread_function_nullary(std::move(task_f)), desc,
                        pika::execution::experimental::get_priority(op_state->scheduler), hint,
                        pika::execution::experimental::get_stacksize(op_state->scheduler));
                    threads::detail::register_work(data, op_state->scheduler.get_thread_pool());
                }

                // Do the work on the worker thread that called set_value
                // from the predecessor sender. This thread participates in
                // the work and does not need a new task since it already
                // runs on a task.
                void do_work_local(std::decay_t<Shape> n, std::uint32_t chunk_size,
                    std::uint32_t worker_thread) const
                {
                    task_function{this->op_state, n, chunk_size, worker_thread}();
                }

                template <typename... Ts>
                void set_value(Ts&&... ts) && noexcept
                {
                    auto r = std::move(*this);
                    // Don't spawn tasks if there is no work to be done
                    if (r.op_state->shape == 0)
                    {
                        pika::execution::experimental::set_value(
                            std::move(r.op_state->receiver), std::forward<Ts>(ts)...);
                        return;
                    }

                    // Calculate chunk size and number of chunks
                    auto const chunk_size =
                        get_chunk_size(r.op_state->num_worker_threads, r.op_state->shape);
                    auto const num_chunks = (r.op_state->shape + chunk_size - 1) / chunk_size;

                    // Store sent values in the operation state
                    r.op_state->ts.template emplace<std::tuple<std::decay_t<Ts>...>>(
                        std::forward<Ts>(ts)...);

                    // Initialize the queues for all worker threads so that
                    // worker threads can start stealing immediately when
                    // they start.
                    for (std::size_t worker_thread = 0;
                         worker_thread < r.op_state->num_worker_threads; ++worker_thread)
                    {
                        r.init_queue(worker_thread, num_chunks);
                    }

                    // Spawn the worker threads for all except the local queue.
                    auto const local_worker_thread = pika::get_local_worker_thread_num();
                    for (std::size_t worker_thread = 0;
                         worker_thread < r.op_state->num_worker_threads; ++worker_thread)
                    {
                        // The queue for the local thread is handled later
                        // inline.
                        if (worker_thread == local_worker_thread) { continue; }

                        r.do_work_task(r.op_state->shape, chunk_size, worker_thread);
                    }

                    // Handle the queue for the local thread.
                    r.do_work_local(r.op_state->shape, chunk_size, local_worker_thread);
                }

                constexpr pika::execution::experimental::empty_env get_env() const& noexcept
                {
                    return {};
                }
            };

            using operation_state_type =
                pika::execution::experimental::connect_result_t<Sender, bulk_receiver>;

            pika::execution::experimental::thread_pool_scheduler scheduler;
            operation_state_type op_state;
            std::size_t num_worker_threads = scheduler.get_thread_pool()->get_os_thread_count();
            std::vector<pika::concurrency::detail::cache_aligned_data<
                pika::concurrency::detail::contiguous_index_queue<>>>
                queues{num_worker_threads};
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Shape> shape;
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            std::atomic<std::decay_t<Shape>> tasks_remaining{
                static_cast<shape_type>(num_worker_threads)};
            pika::util::detail::prepend_t<value_types<std::tuple, pika::detail::variant>,
                pika::detail::monostate>
                ts;
            std::atomic<bool> exception_thrown{false};
            std::optional<std::exception_ptr> exception;

            template <typename Sender_, typename Shape_, typename F_, typename Receiver_>
            operation_state(pika::execution::experimental::thread_pool_scheduler scheduler,
                Sender_&& sender, Shape_&& shape, F_&& f, Receiver_&& receiver)
              : scheduler(std::move(scheduler))
              , op_state(pika::execution::experimental::connect(
                    std::forward<Sender_>(sender), bulk_receiver{this}))
              , shape(std::forward<Shape_>(shape))
              , f(std::forward<F_>(f))
              , receiver(std::forward<Receiver_>(receiver))
            {
            }

            void start() & noexcept { pika::execution::experimental::start(op_state); }
        };

    public:
        template <typename Receiver>
        auto connect(Receiver&& receiver) &&
        {
            return operation_state<std::decay_t<Receiver>>{std::move(scheduler), std::move(sender),
                std::move(shape), std::move(f), std::forward<Receiver>(receiver)};
        }

        template <typename Receiver>
        auto connect(Receiver&& receiver) const&
        {
            return operation_state<std::decay_t<Receiver>>{
                scheduler, sender, shape, f, std::forward<Receiver>(receiver)};
        }

        auto get_env() const& noexcept { return pika::execution::experimental::get_env(sender); }
    };
}    // namespace pika::thread_pool_bulk_detail

namespace pika::execution::experimental {
    template <typename Sender, typename Shape, typename F,
        PIKA_CONCEPT_REQUIRES_(std::is_integral_v<std::decay_t<Shape>>)>
    constexpr auto
    tag_invoke(bulk_t, thread_pool_scheduler scheduler, Sender&& sender, Shape&& shape, F&& f)
    {
        return thread_pool_bulk_detail::thread_pool_bulk_sender<std::decay_t<Sender>,
            std::decay_t<Shape>, std::decay_t<F>>{std::move(scheduler),
            std::forward<Sender>(sender), std::forward<Shape>(shape), std::forward<F>(f)};
    }
}    // namespace pika::execution::experimental
