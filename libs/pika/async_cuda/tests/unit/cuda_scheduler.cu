//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/modules/async_cuda.hpp>

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

    {
        auto s = ex::schedule(sched) |
            cu::then_with_cusolver([](cusolverDnHandle_t) {});
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
            cu::then_with_cusolver([](cusolverDnHandle_t) {}) |
            ex::transfer(ex::thread_pool_scheduler{nullptr});
        CHECK_NOT_CUDA_COMPLETION_SCHEDULER(s);
    }
}
