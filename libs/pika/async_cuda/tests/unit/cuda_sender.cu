//  Copyright (c) 2018 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/cuda.hpp>
#include <pika/execution.hpp>
#include <pika/functional/bind_front.hpp>
#include <pika/init.hpp>
#include <pika/testing.hpp>

#include <whip.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

namespace cu = pika::cuda::experimental;
namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

__global__ void saxpy(int n, float a, float* x, float* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

template <typename Sender>
auto launch_saxpy_kernel(pika::cuda::experimental::cuda_scheduler& cuda_sched, Sender&& predecessor,
    unsigned int& blocks, unsigned int& threads, void** args)
{
    return ex::when_all(std::forward<Sender>(predecessor),
               ex::just(reinterpret_cast<void const*>(&saxpy), dim3(blocks), dim3(threads), args,
                   std::size_t(0))) |
        ex::continues_on(cuda_sched) | cu::then_with_stream(whip::launch_kernel);
}

template <typename T>
__global__ void trivial_kernel(T val)
{
    printf("hello from gpu with value %f\n", static_cast<double>(val));
}

template <typename T>
void cuda_trivial_kernel(T t, whip::stream_t stream)
{
    trivial_kernel<<<1, 1, 0, stream>>>(t);
}

void test_saxpy(pika::cuda::experimental::cuda_scheduler& cuda_sched)
{
    int N = 1 << 20;

    // host arrays (CUDA pinned host memory for asynchronous data transfers)
    float *h_A, *h_B;
    whip::malloc_host(&h_A, N * sizeof(float));
    whip::malloc_host(&h_B, N * sizeof(float));

    // device arrays
    float *d_A, *d_B;
    whip::malloc(&d_A, N * sizeof(float));
    whip::malloc(&d_B, N * sizeof(float));

    // init host data
    for (int idx = 0; idx < N; idx++)
    {
        h_A[idx] = 1.0f;
        h_B[idx] = 2.0f;
    }

    // copy both arrays from cpu to gpu, putting both copies onto the stream
    // no need to get a future back yet
    auto copy_A = ex::just(d_A, h_A, N * sizeof(float), whip::memcpy_host_to_device) |
        ex::continues_on(cuda_sched) | cu::then_with_stream(whip::memcpy_async);
    auto copy_B = ex::just(d_B, h_B, N * sizeof(float), whip::memcpy_host_to_device) |
        ex::continues_on(cuda_sched) | cu::then_with_stream(whip::memcpy_async);

    unsigned int threads = 256;
    unsigned int blocks = (N + 255) / threads;
    float ratio = 2.0f;

    // now launch a kernel on the stream
    void* args[] = {&N, &ratio, &d_A, &d_B};
    auto s = launch_saxpy_kernel(cuda_sched, ex::when_all(std::move(copy_A), std::move(copy_B)),
                 blocks, threads, args) |
        // finally, perform a copy from the gpu back to the cpu all on the same stream
        // grab a future to when this completes
        cu::then_with_stream(pika::util::detail::bind_front(
            whip::memcpy_async, h_B, d_B, N * sizeof(float), whip::memcpy_device_to_host)) |
        // we can add a continuation to the memcpy sender, so that when the
        // memory copy completes, we can do new things ...
        ex::continues_on(ex::thread_pool_scheduler{}) | ex::then([&]() {
            std::cout << "saxpy completed on GPU, checking results in continuation" << std::endl;
            float max_error = 0.0f;
            for (int jdx = 0; jdx < N; jdx++)
            {
                max_error = (std::max)(max_error, abs(h_B[jdx] - 4.0f));
            }
            std::cout << "Max Error: " << max_error << std::endl;

            // We must reach this point. Otherwise something has gone wrong.
            PIKA_TEST(true);
        });

    // the sync_wait() is important because without it, this function returns
    // and the tasks are never spawned.
    tt::sync_wait(std::move(s));
}

// -------------------------------------------------------------------------
int pika_main(pika::program_options::variables_map& vm)
{
    std::size_t device = vm["device"].as<std::size_t>();
    //
    unsigned int seed = (unsigned int) std::time(nullptr);
#if !defined(PIKA_HAVE_HIP)
    // ROCm Clang-15 (HIP 5.3.3) fails to compile this with an internal compiler
    // error. See https://github.com/pika-org/pika/issues/585 for more details.
    if (vm.count("seed")) seed = vm["seed"].as<unsigned int>();
#else
    std::cout << "The --seed command line argument is ignored because HIP is ";
    std::cout << "enabled. See https://github.com/pika-org/pika/issues/585 ";
    std::cout << "for more details." << std::endl;
#endif

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    pika::cuda::experimental::cuda_pool cuda_pool(device);
    // install cuda future polling handler
    pika::cuda::experimental::enable_user_polling poll("default");
    //
    pika::cuda::experimental::cuda_scheduler cuda_sched(std::move(cuda_pool));

    // --------------------
    // test kernel launch<float> using then_with_stream
    float testf = 1.2345;
    std::cout << "schedule : cuda kernel <float>  : " << testf << std::endl;
    tt::sync_wait(ex::just(testf) | ex::continues_on(cuda_sched) |
        cu::then_with_stream(&cuda_trivial_kernel<float>));

    // --------------------
    // test kernel launch<double> using apply and async
    double testd = 1.2345;
    std::cout << "schedule : cuda kernel <double>  : " << testd << std::endl;
    tt::sync_wait(ex::just(testd) | ex::continues_on(cuda_sched) |
        cu::then_with_stream(&cuda_trivial_kernel<double>));

    // --------------------
    // test adding a continuation to a cuda call
    double testd2 = 3.1415;
    std::cout << "then_with_stream/continuation : " << testd2 << std::endl;
    tt::sync_wait(ex::just(testd2) | ex::continues_on(cuda_sched) |
        cu::then_with_stream(&cuda_trivial_kernel<double>));

    // --------------------
    // test using a copy of a cuda executor
    // and adding a continuation with a copy of a copy
    std::cout << "Copying executor : " << testd2 + 1 << std::endl;
    auto cuda_sched_copy = cuda_sched;
    tt::sync_wait(ex::just(testd2 + 1) | ex::continues_on(cuda_sched_copy) |
        cu::then_with_stream(&cuda_trivial_kernel<double>));

    // --------------------
    // test a full kernel example
    test_saxpy(cuda_sched);

    pika::finalize();
    return EXIT_SUCCESS;
}

// -------------------------------------------------------------------------
int main(int argc, char** argv)
{
    printf("[pika Cuda future] - Starting...\n");

    using namespace pika::program_options;
    options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");
    cmdline.add_options()("device", pika::program_options::value<std::size_t>()->default_value(0),
        "Device to use")("iterations",
        pika::program_options::value<std::size_t>()->default_value(30),
        "iterations")("seed,s", pika::program_options::value<unsigned int>(),
        "the random number generator seed to use for this run");

    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return pika::init(pika_main, argc, argv, init_args);
}
