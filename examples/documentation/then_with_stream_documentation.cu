//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/cuda.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>

#include <whip.hpp>

#include <cstddef>
#include <cstdio>
#include <utility>

__global__ void kernel(int* p, int offset)
{
    printf(
        "Hello from kernel! threadIdx.x: %d\n", static_cast<int>(threadIdx.x));
    p[threadIdx.x] = threadIdx.x * threadIdx.x + offset;
}

int main(int argc, char* argv[])
{
    namespace cu = pika::cuda::experimental;
    namespace ex = pika::execution::experimental;
    namespace tt = pika::this_thread::experimental;

    pika::start(argc, argv);
    ex::thread_pool_scheduler cpu_sched{};
    cu::cuda_pool pool{};
    cu::cuda_scheduler cuda_sched{pool};

    {
        cu::enable_user_polling p{};

        constexpr std::size_t n = 32;
        int* a = nullptr;

        // whip::malloc_async wraps cudaMallocAsync/hipMallocAsync. Using the
        // sender adaptors the allocation, work, and deallocation can all be
        // scheduled onto the same stream.
        auto s = ex::just(&a, n * sizeof(int)) | ex::continues_on(cuda_sched) |
            cu::then_with_stream(whip::malloc_async) |
            // The then_with_stream callable accepts values sent by the
            // predecessor. They will be passed by reference before the stream.
            // This allows e.g. whip::malloc_async to be used above with values
            // sent by the just sender. The values are passed by reference and
            // will be kept alive until the work done on the stream is done.
            cu::then_with_stream(
                [&a](
                    /* other values by reference here */ whip::stream_t
                        stream) {
                    kernel<<<1, n, 0, stream>>>(a, 17);
                    // Even though the function returns here, the sync_wait below
                    // will wait for the kernel to finish. Values returned are
                    // passed on to continuations.
                    return a;
                }) |
            cu::then_with_stream(whip::free_async);

        tt::sync_wait(std::move(s));
    }

    pika::finalize();
    pika::stop();

    return 0;
}
