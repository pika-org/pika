//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/cuda.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/testing.hpp>

#include <whip.hpp>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <functional>
#include <string>
#include <utility>

namespace cu = pika::cuda::experimental;
namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

int main(int argc, char* argv[])
{
    pika::start(argc, argv);

    {
        cu::cuda_pool pool{};
        cu::enable_user_polling p;

        // cuBLAS handle on a non-pika thread
        tt::sync_wait(ex::schedule(cu::cuda_scheduler{pool}) |
            cu::then_with_cublas([](cublasHandle_t) {}, CUBLAS_POINTER_MODE_HOST));

        // cuBLAS handle on a pika thread
        tt::sync_wait(ex::schedule(ex::thread_pool_scheduler{}) |
            ex::transfer(cu::cuda_scheduler{pool}) |
            cu::then_with_cublas([](cublasHandle_t) {}, CUBLAS_POINTER_MODE_HOST));

        // cuSOLVER handle on a non-pika thread
        tt::sync_wait(ex::schedule(cu::cuda_scheduler{pool}) |
            cu::then_with_cusolver([](cusolverDnHandle_t) {}));

        // cuSOLVER handle on a pika thread
        tt::sync_wait(ex::schedule(ex::thread_pool_scheduler{}) |
            ex::transfer(cu::cuda_scheduler{pool}) |
            cu::then_with_cusolver([](cusolverDnHandle_t) {}));
    }

    pika::finalize();
    pika::stop();

    whip::check_error(
#if defined(PIKA_HAVE_CUDA)
        cudaDeviceReset()
#elif defined(PIKA_HAVE_HIP)
        hipDeviceReset()
#endif
    );

    PIKA_TEST(true);
}
