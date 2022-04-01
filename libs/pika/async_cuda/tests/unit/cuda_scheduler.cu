//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/cuda.hpp>
#include <pika/execution.hpp>
#include <pika/testing.hpp>

#include <type_traits>

namespace cu = pika::cuda::experimental;
namespace ex = pika::execution::experimental;

template <typename Scheduler>
inline constexpr bool is_cuda_scheduler_v =
    std::is_same_v<std::decay_t<Scheduler>, cu::cuda_scheduler>;

#define CHECK_CUDA_COMPLETION_SCHEDULER(...)                                   \
    static_assert(is_cuda_scheduler_v<decltype(                                \
            ex::get_completion_scheduler<ex::set_value_t>(__VA_ARGS__))>)

#define CHECK_NOT_CUDA_COMPLETION_SCHEDULER(...)                               \
    static_assert(                                                             \
        !is_cuda_scheduler_v<decltype(                                         \
            ex::get_completion_scheduler<ex::set_value_t>(__VA_ARGS__))>)

int main()
{
    cu::cuda_pool pool{};
    cu::cuda_scheduler sched{pool};

    // Check that the completion scheduler is correctly set for various senders
    {
        auto s = ex::schedule(sched);
        CHECK_CUDA_COMPLETION_SCHEDULER(s);
    }

    {
        auto s = ex::just() | ex::transfer(sched);
        CHECK_CUDA_COMPLETION_SCHEDULER(s);
    }

    {
        auto s =
            ex::schedule(sched) | cu::then_with_stream([](cudaStream_t) {});
        CHECK_CUDA_COMPLETION_SCHEDULER(s);
    }

    {
        auto s = ex::schedule(sched) |
            cu::then_with_cublas(
                [](cublasHandle_t) {}, CUBLAS_POINTER_MODE_HOST);
        CHECK_CUDA_COMPLETION_SCHEDULER(s);
    }

#if defined(PIKA_WITH_CUDA)
    {
        auto s = ex::schedule(sched) |
            cu::then_with_cusolver([](cusolverDnHandle_t) {});
        CHECK_CUDA_COMPLETION_SCHEDULER(s);
    }
#endif

    {
        auto s = ex::schedule(sched) |
            cu::then_with_any_cuda(
                [](cublasHandle_t) {}, CUBLAS_POINTER_MODE_HOST);
        CHECK_CUDA_COMPLETION_SCHEDULER(s);
    }

    {
        auto s = ex::schedule(sched) | cu::then_on_host([]() {});
        CHECK_CUDA_COMPLETION_SCHEDULER(s);
    }

    {
        // This test initializes the thread_pool_scheduler with nullptr only to
        // avoid it trying to get a thread pool through the default thread pool
        // handler which is not installed in this test (the HPX runtime is not
        // started). The thread pool is never accessed.
        auto s = ex::schedule(sched) |
            cu::then_with_cublas(
                [](cublasHandle_t) {}, CUBLAS_POINTER_MODE_HOST) |
            ex::transfer(ex::thread_pool_scheduler{nullptr});
        CHECK_NOT_CUDA_COMPLETION_SCHEDULER(s);
    }

    {
        cu::cuda_scheduler sched{pool};

        // This partly tests implementation details. The scheduler is not
        // guaranteed to return a stream with the exact same priority as given
        // to the scheduler. It will return a stream with a priority "close to"
        // the given priority. Currently this means that anything high or higher
        // maps to high, and anything below high maps to normal.
        PIKA_TEST_EQ(sched.get_next_stream().get_priority(),
            pika::threads::thread_priority::normal);
        PIKA_TEST_EQ(
            ex::with_priority(sched, pika::threads::thread_priority::low)
                .get_next_stream()
                .get_priority(),
            pika::threads::thread_priority::normal);
        PIKA_TEST_EQ(
            ex::with_priority(sched, pika::threads::thread_priority::default_)
                .get_next_stream()
                .get_priority(),
            pika::threads::thread_priority::normal);
        PIKA_TEST_EQ(
            ex::with_priority(sched, pika::threads::thread_priority::high)
                .get_next_stream()
                .get_priority(),
            pika::threads::thread_priority::high);
    }

    {
        PIKA_TEST(ex::get_forward_progress_guarantee(sched) ==
            ex::forward_progress_guarantee::weakly_parallel);
    }
}
