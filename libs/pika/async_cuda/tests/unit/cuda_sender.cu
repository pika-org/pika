//  Copyright (c) 2018 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/algorithm.hpp>
#include <pika/cuda.hpp>
#include <pika/execution.hpp>
#include <pika/functional.hpp>
#include <pika/init.hpp>
#include <pika/testing.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

namespace cu = pika::cuda::experimental;
namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

constexpr auto cuda_launch_kernel = [](auto&&... ts) {
    cu::check_cuda_error(cudaLaunchKernel(std::forward<decltype(ts)>(ts)...));
};

constexpr auto cuda_memcpy_async = [](auto&&... ts) {
    cu::check_cuda_error(cudaMemcpyAsync(std::forward<decltype(ts)>(ts)...));
};

__global__ void saxpy(int n, float a, float* x, float* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

template <typename Sender>
auto launch_saxpy_kernel(pika::cuda::experimental::cuda_scheduler& cuda_sched,
    Sender&& predecessor, unsigned int& blocks, unsigned int& threads,
    void** args)
{
    return ex::when_all(std::forward<Sender>(predecessor),
               ex::just(reinterpret_cast<const void*>(&saxpy), dim3(blocks),
                   dim3(threads), args, std::size_t(0))) |
        ex::transfer(cuda_sched) | cu::then_with_stream(cuda_launch_kernel);
}

template <typename T>
__global__ void trivial_kernel(T val)
{
    // TODO: Fingers crossed that printf now works with HIP?
    printf("hello from gpu with value %f\n", static_cast<double>(val));
}

template <typename T>
void cuda_trivial_kernel(T t, cudaStream_t stream)
{
    trivial_kernel<<<1, 1, 0, stream>>>(t);
}

void test_saxpy(pika::cuda::experimental::cuda_scheduler& cuda_sched)
{
    int N = 1 << 20;

    // host arrays (CUDA pinned host memory for asynchronous data transfers)
    float *h_A, *h_B;
    pika::cuda::experimental::check_cuda_error(
        cudaMallocHost((void**) &h_A, N * sizeof(float)));
    pika::cuda::experimental::check_cuda_error(
        cudaMallocHost((void**) &h_B, N * sizeof(float)));

    // device arrays
    float *d_A, *d_B;
    pika::cuda::experimental::check_cuda_error(
        cudaMalloc((void**) &d_A, N * sizeof(float)));

    pika::cuda::experimental::check_cuda_error(
        cudaMalloc((void**) &d_B, N * sizeof(float)));

    // init host data
    for (int idx = 0; idx < N; idx++)
    {
        h_A[idx] = 1.0f;
        h_B[idx] = 2.0f;
    }

    // copy both arrays from cpu to gpu, putting both copies onto the stream
    // no need to get a future back yet
    auto copy_A = ex::transfer_just(cuda_sched, d_A, h_A, N * sizeof(float),
                      cudaMemcpyHostToDevice) |
        cu::then_with_stream(cuda_memcpy_async);
    auto copy_B = ex::transfer_just(cuda_sched, d_B, h_B, N * sizeof(float),
                      cudaMemcpyHostToDevice) |
        cu::then_with_stream(cuda_memcpy_async);

    unsigned int threads = 256;
    unsigned int blocks = (N + 255) / threads;
    float ratio = 2.0f;

    // now launch a kernel on the stream
    void* args[] = {&N, &ratio, &d_A, &d_B};
    auto s = launch_saxpy_kernel(cuda_sched,
                 ex::when_all(std::move(copy_A), std::move(copy_B)), blocks,
                 threads, args) |
        // finally, perform a copy from the gpu back to the cpu all on the same stream
        // grab a future to when this completes
        cu::then_with_stream(pika::bind_front(cuda_memcpy_async, h_B, d_B,
            N * sizeof(float), cudaMemcpyDeviceToHost)) |
        // we can add a continuation to the memcpy sender, so that when the
        // memory copy completes, we can do new things ...
        ex::transfer(ex::thread_pool_scheduler{}) | ex::then([&]() {
            std::cout
                << "saxpy completed on GPU, checking results in continuation"
                << std::endl;
            float max_error = 0.0f;
            for (int jdx = 0; jdx < N; jdx++)
            {
                max_error = (std::max)(max_error, abs(h_B[jdx] - 4.0f));
            }
            std::cout << "Max Error: " << max_error << std::endl;
        });

    // the sync_wait() is important because without it, this function returns
    // and the tasks are never spawned.
    tt::sync_wait(std::move(s));
}

// -------------------------------------------------------------------------
int pika_main(pika::program_options::variables_map& vm)
{
    // install cuda future polling handler
    pika::cuda::experimental::enable_user_polling poll("default");
    //
    std::size_t device = vm["device"].as<std::size_t>();
    //
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    pika::cuda::experimental::cuda_pool cuda_pool(device);
    pika::cuda::experimental::cuda_scheduler cuda_sched(std::move(cuda_pool));

    // --------------------
    // test kernel launch<float> using then_with_stream
    float testf = 1.2345;
    std::cout << "schedule : cuda kernel <float>  : " << testf << std::endl;
    tt::sync_wait(ex::transfer_just(cuda_sched, testf) |
        cu::then_with_stream(&cuda_trivial_kernel<float>));

    // --------------------
    // test kernel launch<double> using apply and async
    float testd = 1.2345;
    std::cout << "schedule : cuda kernel <double>  : " << testf << std::endl;
    tt::sync_wait(ex::transfer_just(cuda_sched, testd) |
        cu::then_with_stream(&cuda_trivial_kernel<double>));

    // --------------------
    // test adding a continuation to a cuda call
    double testd2 = 3.1415;
    std::cout << "then_with_stream/continuation : " << testd2 << std::endl;
    tt::sync_wait(ex::transfer_just(cuda_sched, testd2) |
        cu::then_with_stream(&cuda_trivial_kernel<double>) |
        cu::then_on_host([] { std::cout << "continuation triggered\n"; }));

    // --------------------
    // test using a copy of a cuda executor
    // and adding a continuation with a copy of a copy
    std::cout << "Copying executor : " << testd2 + 1 << std::endl;
    auto cuda_sched_copy = cuda_sched;
    tt::sync_wait(ex::transfer_just(cuda_sched, testd2 + 1) |
        cu::then_with_stream(&cuda_trivial_kernel<double>) |
        cu::then_on_host([] { std::cout << "copy continuation triggered\n"; }));

    // --------------------
    // test a full kernel example
    test_saxpy(cuda_sched);

    return pika::finalize();
}

// -------------------------------------------------------------------------
int main(int argc, char** argv)
{
    printf("[pika Cuda future] - Starting...\n");

    using namespace pika::program_options;
    options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");
    cmdline.add_options()("device",
        pika::program_options::value<std::size_t>()->default_value(0),
        "Device to use")("iterations",
        pika::program_options::value<std::size_t>()->default_value(30),
        "iterations")("seed,s", pika::program_options::value<unsigned int>(),
        "the random number generator seed to use for this run");

    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;

    auto result = pika::init(pika_main, argc, argv, init_args);
    return result || pika::util::report_errors();
}
