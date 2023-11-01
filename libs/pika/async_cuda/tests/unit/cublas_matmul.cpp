//  Copyright (c) 2017-2018 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// For compliance with the NVIDIA EULA:
// "This software contains source code provided by NVIDIA Corporation."

// This is a conversion of the NVIDIA cublas example matrixMulCUBLAS to use
// pika style data structures, executors and futures and demonstrate a simple use
// of computing a number of iteration of a matrix multiply on a stream and returning
// a future when it completes. This can be used to chain/schedule other task
// in a manner consistent with the future based API of pika.
//
// Example usage: bin/cublas_matmul --sizemult=10 --iterations=25 --pika:threads=8
// NB. The pika::threads param only controls how many parallel tasks to use for the CPU
// comparison/checks and makes no difference to the GPU execution.
//
// Note: The pika::cuda::experimental::allocator makes use of device code and if used
// this example must be compiled with nvcc instead of c++ which requires the following
// cmake setting
// set_source_files_properties(cublas_matmul.cpp
//     PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
// Currently, nvcc does not handle lambda functions properly and it is simpler to use
// cudaMalloc/cudaMemcpy etc, so we do not #define PIKA_CUBLAS_DEMO_WITH_ALLOCATOR

#include <pika/assert.hpp>
#include <pika/chrono.hpp>
#include <pika/cuda.hpp>
#include <pika/execution.hpp>
#include <pika/functional/bind_front.hpp>
#include <pika/init.hpp>
#include <pika/testing.hpp>

#include <whip.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
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
// Compute reference data set matrix multiply on CPU
// C = A * B
// @param C          reference data, computed but preallocated
// @param A          matrix A as provided to device
// @param B          matrix B as provided to device
// @param hA         height of matrix A
// @param wB         width of matrix B
// -------------------------------------------------------------------------
template <typename T>
void matrixMulCPU(T* C, const T* A, const T* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
    {
        for (unsigned int j = 0; j < wB; ++j)
        {
            T sum = 0;
            for (unsigned int k = 0; k < wA; ++k)
            {
                T a = A[i * wA + k];
                T b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (T) sum;
        }
    }
}

// -------------------------------------------------------------------------
// Compute the L2 norm difference between two arrays
inline bool compare_L2_err(
    const float* reference, const float* data, const unsigned int len, const float epsilon)
{
    PIKA_ASSERT(epsilon >= 0);

    float error = 0;
    float ref = 0;

    for (unsigned int i = 0; i < len; ++i)
    {
        float diff = reference[i] - data[i];
        error += diff * diff;
        ref += reference[i] * reference[i];
    }

    float normRef = sqrtf(ref);
    if (std::fabs(ref) < 1e-7f) { return false; }

    float normError = sqrtf(error);
    error = normError / normRef;
    bool result = error < epsilon;
    return result;
}

// -------------------------------------------------------------------------
// Run a simple test matrix multiply using CUBLAS
// -------------------------------------------------------------------------
template <typename T>
void matrixMultiply(pika::cuda::experimental::cuda_scheduler& cuda_sched, sMatrixSize& matrix_size,
    std::size_t /* device */, [[maybe_unused]] std::size_t iterations)
{
    // Allocate host memory for matrices A and B
    unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;

    std::vector<T> h_A(size_A);
    std::vector<T> h_B(size_B);
    std::vector<T> h_C(size_C);
    std::vector<T> h_CUBLAS(size_C);

    // Fill A and B with random numbers
    auto randfunc = [](T& x) { x = gen() / (T) RAND_MAX; };
    std::for_each(h_A.begin(), h_A.end(), randfunc);
    std::for_each(h_B.begin(), h_B.end(), randfunc);

    T *d_A, *d_B, *d_C;
    whip::malloc(&d_A, size_A * sizeof(T));
    whip::malloc(&d_B, size_B * sizeof(T));
    whip::malloc(&d_C, size_C * sizeof(T));

    auto copy_A = ex::schedule(cuda_sched) |
        cu::then_with_stream(pika::util::detail::bind_front(
            whip::memcpy_async, d_A, h_A.data(), size_A * sizeof(T), whip::memcpy_host_to_device));
    auto copy_B = ex::schedule(cuda_sched) |
        cu::then_with_stream(pika::util::detail::bind_front(
            whip::memcpy_async, d_B, h_B.data(), size_B * sizeof(T), whip::memcpy_host_to_device));
    auto copy_AB = ex::when_all(std::move(copy_A), std::move(copy_B)) | ex::then([]() {
        std::cout << "The async host->device copy operations completed" << std::endl;
    });
    tt::sync_wait(std::move(copy_AB));

    std::cout << "Computing result using CUBLAS...\n";
    const T alpha = 1.0f;
    const T beta = 0.0f;

    // Perform warmup operation with cublas
    // note cublas is column major ordering : transpose the order
    pika::chrono::detail::high_resolution_timer t1;
    //
    std::cout << "calling CUBLAS...\n";
    auto gemm = ex::transfer_just(cuda_sched) |
        cu::then_with_cublas(
            [&](cublasHandle_t handle) {
                cu::check_cublas_error(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B,
                    matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWA));
            },
            CUBLAS_POINTER_MODE_HOST);

    // wait until the operation completes
    tt::sync_wait(std::move(gemm));

    double us1 = t1.elapsed<std::chrono::microseconds>();
    std::cout << "warmup: elapsed_microseconds " << us1 << std::endl;

    // once the sender has been synchronized, the next call to
    // schedule/then_with_x will create a new event attached to a new sender so
    // we can reuse the same cuda scheduler stream if we want

    // See https://github.com/brycelelbach/wg21_p2300_std_execution/issues/466
    // for details.
#if defined(PIKA_HAVE_STDEXEC)
    std::cout << "skipping remainder of test because the stdexec implementation of split does not "
                 "yet support move-only senders"
              << std::endl;
#else
    pika::chrono::detail::high_resolution_timer t2;

    // This loop is currently inefficient. Because of the type-erasure with
    // unique_any_sender the cuBLAS calls are not scheduled on the same stream
    // without synchronization.
    ex::unique_any_sender<> gemms_finished = ex::just();
    for (std::size_t j = 0; j < iterations; j++)
    {
        gemms_finished = std::move(gemms_finished) | ex::transfer(cuda_sched) |
            cu::then_with_cublas(
                [&](cublasHandle_t handle) {
                    cu::check_cublas_error(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B,
                        matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWA));
                },
                CUBLAS_POINTER_MODE_HOST);
    }

    auto gemms_finished_split = ex::split(std::move(gemms_finished));

    auto matrix_print_finished = gemms_finished_split | ex::then([&]() {
        double us2 = t2.elapsed<std::chrono::microseconds>();
        std::cout << "actual: elapsed_microseconds " << us2 << " iterations " << iterations
                  << std::endl;

        // Compute and print the performance
        double usecPerMatrixMul = us2 / iterations;
        double flopsPerMatrixMul =
            2.0 * (double) matrix_size.uiWA * (double) matrix_size.uiHA * (double) matrix_size.uiWB;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-9) / (usecPerMatrixMul / 1e6);
        printf("Performance = %.2f GFlop/s, Time = %.3f msec/iter, Size = %.0f Ops\n", gigaFlops,
            1e-3 * usecPerMatrixMul, flopsPerMatrixMul);
    });

    // when the matrix operations complete, copy the result to the host
    auto copy_finished = std::move(gemms_finished_split) | ex::transfer(cuda_sched) |
        cu::then_with_stream(pika::util::detail::bind_front(whip::memcpy_async, h_CUBLAS.data(),
            d_C, size_C * sizeof(T), whip::memcpy_device_to_host));

    auto all_done =
        ex::when_all(std::move(matrix_print_finished), std::move(copy_finished)) | ex::then([&]() {
            // compute reference solution on the CPU
            std::cout << "\nComputing result using host CPU...\n";

            // compute reference solution on the CPU
            // allocate storage for the CPU result
            std::vector<T> reference(size_C);

            pika::chrono::detail::high_resolution_timer t3;
            matrixMulCPU<T>(reference.data(), h_A.data(), h_B.data(), matrix_size.uiHA,
                matrix_size.uiWA, matrix_size.uiWB);
            double us3 = t3.elapsed<std::chrono::microseconds>();
            std::cout << "CPU elapsed_microseconds (1 iteration) " << us3 << std::endl;

            // check result (CUBLAS)
            bool resCUBLAS = compare_L2_err(reference.data(), h_CUBLAS.data(), size_C, 1e-6);
            PIKA_TEST_MSG(resCUBLAS, "matrix CPU/GPU comparison error");

            // if the result was incorrect, we throw an exception, so here it's ok
            if (resCUBLAS)
            {
                std::cout << "\nComparing CUBLAS Matrix Multiply with CPU results: OK \n";
            }
        });

    tt::sync_wait(std::move(all_done));
#endif
    whip::free(d_A);
    whip::free(d_B);
    whip::free(d_C);
}

// -------------------------------------------------------------------------
int pika_main(pika::program_options::variables_map& vm)
{
    //
    std::size_t device = vm["device"].as<std::size_t>();
    std::size_t sizeMult = vm["sizemult"].as<std::size_t>();
    std::size_t iterations = vm["iterations"].as<std::size_t>();
    //
    unsigned int seed = std::random_device{}();
    if (vm.count("seed")) seed = vm["seed"].as<unsigned int>();

    pika::cuda::experimental::cuda_pool cuda_pool(device);

    // install cuda future polling handler
    pika::cuda::experimental::enable_user_polling poll("default");
    //

    gen.seed(seed);
    std::cout << "using seed: " << seed << std::endl;

    //
    sizeMult = (std::min)(sizeMult, std::size_t(100));
    sizeMult = (std::max)(sizeMult, std::size_t(1));
    //
    int block_size = 32;

    sMatrixSize matrix_size;
    matrix_size.uiWA = 2 * block_size * sizeMult;
    matrix_size.uiHA = 4 * block_size * sizeMult;
    matrix_size.uiWB = 2 * block_size * sizeMult;
    matrix_size.uiHB = 4 * block_size * sizeMult;
    matrix_size.uiWC = 2 * block_size * sizeMult;
    matrix_size.uiHC = 4 * block_size * sizeMult;

    printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n\n", matrix_size.uiWA, matrix_size.uiHA,
        matrix_size.uiWB, matrix_size.uiHB, matrix_size.uiWC, matrix_size.uiHC);

    // --------------------------------
    // test matrix multiply using cuda scheduler
    pika::cuda::experimental::cuda_scheduler cuda_sched(std::move(cuda_pool));
    matrixMultiply<float>(cuda_sched, matrix_size, device, iterations);

    // --------------------------------
    // sanity check : test again using a copy of the cuda scheduler
    std::cout << "\n\n\n------------" << std::endl;
    std::cout << "Checking copy semantics of cuda scheduler" << std::endl;
    pika::cuda::experimental::cuda_scheduler cuda_sched2 = cuda_sched;
    matrixMultiply<float>(cuda_sched2, matrix_size, device, 1);

    // --------------------------------
    // sanity check : test again using a moved copy of the cuda scheduler
    std::cout << "\n\n\n------------" << std::endl;
    std::cout << "Checking move semantics of cuda scheduler" << std::endl;
    pika::cuda::experimental::cuda_scheduler cuda_sched3(std::move(cuda_sched));
    matrixMultiply<float>(cuda_sched3, matrix_size, device, 1);

    pika::finalize();
    return EXIT_SUCCESS;
}

// -------------------------------------------------------------------------
int main(int argc, char** argv)
{
    printf("[pika Matrix Multiply CUBLAS] - Starting...\n");

    using namespace pika::program_options;
    options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");
    // clang-format off
    cmdline.add_options()
        ("device",
        pika::program_options::value<std::size_t>()->default_value(0),
        "Device to use")
        ("sizemult",
        pika::program_options::value<std::size_t>()->default_value(5),
        "Multiplier")
        ("iterations",
        pika::program_options::value<std::size_t>()->default_value(30),
        "iterations")
        ("no-cpu",
        pika::program_options::value<bool>()->default_value(false),
        "disable CPU validation to save time")
        ("seed,s",
        pika::program_options::value<unsigned int>(),
        "the random number generator seed to use for this run");
    // clang-format on
    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;

#if defined(PIKA_HAVE_STDEXEC)
    PIKA_TEST(true);
#endif

    return pika::init(pika_main, argc, argv, init_args);
}
