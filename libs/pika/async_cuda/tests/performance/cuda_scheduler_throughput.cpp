//  Copyright (c) 2017-2018 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// For compliance with the NVIDIA EULA:
// "This software contains source code provided by NVIDIA Corporation."

// This is a conversion of the NVIDIA cublas example matrixMulCUBLAS to use pika
// style data structures, schedulers and senders and demonstrate a simple use of
// computing a number of iteration of a matrix multiply on a stream and
// returning a sender when it completes. This can be used to chain/schedule
// other task in a manner consistent with the sender based API of pika.
//
// Example usage: bin/cublas_matmul --sizemult=10 --iterations=25
// --pika:threads=8 NB. The pika::threads param only controls how many parallel
// tasks to use for the CPU comparison/checks and makes no difference to the GPU
// execution.

#include <pika/cuda.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/modules/timing.hpp>
#include <pika/testing/performance.hpp>

#include <whip.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(PIKA_HAVE_HIP)
# define CUBLAS_OP_N rocblas_operation_none
# define cublasSgemm rocblas_sgemm
#endif

std::mt19937 gen;

namespace cu = pika::cuda::experimental;
namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

// -------------------------------------------------------------------------
// Optional Command-line multiplier for matrix sizes
struct sMatrixSize
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
};

// -------------------------------------------------------------------------
// Run a simple test matrix multiply using CUBLAS
// -------------------------------------------------------------------------
template <typename T>
void matrixMultiply(sMatrixSize& matrix_size, std::size_t device, std::size_t iterations)
{
    using pika::execution::par;

    // Allocate host memory for matrices A and B
    unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;

    std::vector<T> h_A(size_A);
    std::vector<T> h_B(size_B);
    std::vector<T> h_C(size_C);
    std::vector<T> h_CUBLAS(size_C);

    // Fill A and B with zeroes
    auto zerofunc = [](T& x) { x = 0; };
    std::for_each(h_A.begin(), h_A.end(), zerofunc);
    std::for_each(h_B.begin(), h_B.end(), zerofunc);

    // create a cuda executor we'll use to schedule cuda work
    cu::cuda_pool cuda_pool(device);
    cu::cuda_scheduler cuda_sched(std::move(cuda_pool));

    // install cuda future polling handler for this scope
    cu::enable_user_polling poll("default");

    T *d_A, *d_B, *d_C;
    whip::malloc(&d_A, size_A * sizeof(T));
    whip::malloc(&d_B, size_B * sizeof(T));
    whip::malloc(&d_C, size_C * sizeof(T));

    // copy A and B to device
    auto copies_done = ex::when_all(ex::transfer_just(cuda_sched, d_A, h_A.data(),
                                        size_A * sizeof(T), whip::memcpy_host_to_device) |
            cu::then_with_stream(whip::memcpy_async),
        ex::transfer_just(
            cuda_sched, d_B, h_B.data(), size_B * sizeof(T), whip::memcpy_host_to_device) |
            cu::then_with_stream(whip::memcpy_async));

    // print something when copies complete
    tt::sync_wait(std::move(copies_done) | ex::then([] {
        std::cout << "Async host->device copy operation completed" << std::endl << std::endl;
    }));

    std::cout << "Small matrix multiply tests using CUBLAS...\n\n";
    const T alpha = 1.0f;
    const T beta = 0.0f;

    auto test_function = [&](cu::cuda_scheduler& cuda_sched, const std::string& msg,
                             std::size_t n_iters) {
        // time many cuda kernels spawned one after each other when they complete
        pika::chrono::detail::high_resolution_timer t1;
        for (std::size_t j = 0; j < n_iters; j++)
        {
            tt::sync_wait(ex::schedule(cuda_sched) |
                cu::then_with_cublas(
                    [&](cublasHandle_t handle) {
                        cu::check_cublas_error(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B,
                            matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWA));
                    },
                    CUBLAS_POINTER_MODE_HOST));
        }
        double us1 = t1.elapsed<std::chrono::microseconds>();
        std::cout << "us per iteration " << us1 / n_iters << " : " << msg << std::endl << std::endl;
    };

    test_function(cuda_sched, "Event polling based scheduler", iterations);

    whip::free(d_A);
    whip::free(d_B);
    whip::free(d_C);
}

// -------------------------------------------------------------------------
int pika_main(pika::program_options::variables_map& vm)
{
    std::size_t device = vm["device"].as<std::size_t>();
    std::size_t iterations = vm["iterations"].as<std::size_t>();
    //
    int sizeMult = 1;

    int block_size = 4;
    sMatrixSize matrix_size;
    matrix_size.uiWA = 1 * block_size * sizeMult;
    matrix_size.uiHA = 1 * block_size * sizeMult;
    matrix_size.uiWB = 1 * block_size * sizeMult;
    matrix_size.uiHB = 1 * block_size * sizeMult;
    matrix_size.uiWC = 1 * block_size * sizeMult;
    matrix_size.uiHC = 1 * block_size * sizeMult;

    printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n\n", matrix_size.uiWA, matrix_size.uiHA,
        matrix_size.uiWB, matrix_size.uiHB, matrix_size.uiWC, matrix_size.uiHC);

    matrixMultiply<float>(matrix_size, device, iterations);

    return pika::finalize();
}

// -------------------------------------------------------------------------
int main(int argc, char** argv)
{
    printf("[pika CUDA scheduler executor benchmark] - Starting...\n");

    using namespace pika::program_options;
    options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");
    // clang-format off
    cmdline.add_options()
        ("device",
        pika::program_options::value<std::size_t>()->default_value(0),
        "Device to use")
        ("iterations",
        pika::program_options::value<std::size_t>()->default_value(30),
        "iterations");
    // clang-format on

    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return pika::init(pika_main, argc, argv, init_args);
}
