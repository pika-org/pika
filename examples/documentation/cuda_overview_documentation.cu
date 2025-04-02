//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/cuda.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>

#include <whip.hpp>

#include <cstdio>
#include <utility>

__global__ void kernel()
{
    printf(
        "Hello from kernel! threadIdx.x: %d\n", static_cast<int>(threadIdx.x));
}

int main(int argc, char* argv[])
{
    namespace cu = pika::cuda::experimental;
    namespace ex = pika::execution::experimental;
    namespace tt = pika::this_thread::experimental;

    pika::start(argc, argv);
    ex::thread_pool_scheduler cpu_sched{};

    // Create a pool of CUDA streams and cuBLAS/SOLVER handles, and a scheduler
    // that uses the pool.
    cu::cuda_pool pool{};
    cu::cuda_scheduler cuda_sched{pool};

    {
        // Enable polling of CUDA events on the default pool. This is required
        // to allow the adaptors below to signal completion of kernels.
        cu::enable_user_polling p{};

        // The work created by the adaptors below will all be scheduled on the
        // same stream from the pool since the work is sequential.
        //
        // Note that error checking is omitted below.
        auto s = ex::just(42) | ex::continues_on(cuda_sched) |
            // CUDA kernel through a lambda.
            ex::then([](int x) { printf("Hello from the GPU! x: %d\n", x); }) |
            // Explicitly launch a CUDA kernel with a stream (see
            // https://github.com/eth-cscs/whip for details about whip)
            cu::then_with_stream(
                [](whip::stream_t stream) { kernel<<<1, 32, 0, stream>>>(); });
        tt::sync_wait(std::move(s));
    }

    pika::finalize();
    pika::stop();

    return 0;
}
