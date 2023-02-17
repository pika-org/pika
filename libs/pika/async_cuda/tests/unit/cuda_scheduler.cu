//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/cuda.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/testing.hpp>

#include <whip.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace cu = pika::cuda::experimental;
namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

__global__ void kernel(int* p, int i)
{
    p[i] = i * 2;
}

template <typename Scheduler>
inline constexpr bool is_cuda_scheduler_v =
    std::is_same_v<std::decay_t<Scheduler>, cu::cuda_scheduler>;

#define CHECK_CUDA_COMPLETION_SCHEDULER(...)                                                       \
 static_assert(                                                                                    \
     is_cuda_scheduler_v<decltype(ex::get_completion_scheduler<ex::set_value_t>(__VA_ARGS__))>)

#define CHECK_NOT_CUDA_COMPLETION_SCHEDULER(...)                                                   \
 static_assert(                                                                                    \
     !is_cuda_scheduler_v<decltype(ex::get_completion_scheduler<ex::set_value_t>(__VA_ARGS__))>)

int pika_main()
{
    pika::scoped_finalize sf;

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
        auto s = ex::schedule(sched) | cu::then_with_stream([](whip::stream_t) {});
        CHECK_CUDA_COMPLETION_SCHEDULER(s);
    }

    {
        auto s = ex::schedule(sched) |
            cu::then_with_cublas([](cublasHandle_t) {}, CUBLAS_POINTER_MODE_HOST);
        CHECK_CUDA_COMPLETION_SCHEDULER(s);
    }

    {
        auto s = ex::schedule(sched) | cu::then_with_cusolver([](cusolverDnHandle_t) {});
        CHECK_CUDA_COMPLETION_SCHEDULER(s);
    }

    {
        auto s = ex::schedule(sched) | cu::then_on_host([]() {});
        CHECK_CUDA_COMPLETION_SCHEDULER(s);
    }

    {
#if !defined(PIKA_HAVE_CUDA) || defined(PIKA_CLANG_VERSION)
        // This test initializes the thread_pool_scheduler with nullptr only to
        // avoid it trying to get a thread pool through the default thread pool
        // handler which is not installed in this test (the pika runtime is not
        // started). The thread pool is never accessed.
        auto s = ex::schedule(sched) |
            cu::then_with_cublas([](cublasHandle_t) {}, CUBLAS_POINTER_MODE_HOST) |
            ex::transfer(ex::thread_pool_scheduler{nullptr});
        CHECK_NOT_CUDA_COMPLETION_SCHEDULER(s);
#endif
    }

    {
        cu::cuda_scheduler sched{pool};

        // This partly tests implementation details. The scheduler is not
        // guaranteed to return a stream with the exact same priority as given
        // to the scheduler. It will return a stream with a priority "close to"
        // the given priority. Currently this means that anything high or higher
        // maps to high, and anything below high maps to normal.
        PIKA_TEST_EQ(
            sched.get_next_stream().get_priority(), pika::execution::thread_priority::normal);
        PIKA_TEST_EQ(ex::with_priority(sched, pika::execution::thread_priority::low)
                         .get_next_stream()
                         .get_priority(),
            pika::execution::thread_priority::normal);
        PIKA_TEST_EQ(ex::with_priority(sched, pika::execution::thread_priority::default_)
                         .get_next_stream()
                         .get_priority(),
            pika::execution::thread_priority::normal);
        PIKA_TEST_EQ(ex::with_priority(sched, pika::execution::thread_priority::high)
                         .get_next_stream()
                         .get_priority(),
            pika::execution::thread_priority::high);
    }

    {
        PIKA_TEST(ex::get_forward_progress_guarantee(sched) ==
            ex::forward_progress_guarantee::weakly_parallel);
    }

    // Schedule work with the scheduler
    {
        cu::enable_user_polling poll("default");

        int const n = 1000;
        int* p;
        whip::malloc(&p, sizeof(int) * n);

        cu::cuda_pool pool{};
        cu::cuda_scheduler sched{pool};

        std::vector<ex::unique_any_sender<>> senders;
        senders.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            using pika::execution::thread_priority;

            senders.push_back(ex::schedule(ex::with_priority(
                                  sched, i % 2 ? thread_priority::high : thread_priority::normal)) |
                cu::then_with_stream([p, i](whip::stream_t stream) {
                    kernel<<<1, 1, 0, stream>>>(p, i);
                    whip::check_last_error();
                }));
        }

        // This should use the following:
        //
        //     tt::sync_wait(ex::when_all_vector(std::move(senders)));
        //
        // However, nvcc fails to compile it with an internal compiler error so
        // we use the less efficient but working manual version of it.
        for (auto& s : senders)
        {
            tt::sync_wait(std::move(s));
        }

        std::vector<int> s(n, 0);

        whip::memcpy(s.data(), p, sizeof(int) * n, whip::memcpy_device_to_host);
        whip::free(p);

        for (int i = 0; i < n; ++i)
        {
            PIKA_TEST_EQ(s[i], i * 2);
        }
    }

    return 0;
}

int main(int argc, char** argv)
{
    PIKA_TEST_EQ(pika::init(pika_main, argc, argv), 0);
    return 0;
}
