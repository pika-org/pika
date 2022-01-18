//  Copyright (c) 2017 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/chrono.hpp>
#include <pika/local/execution.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/async_cuda.hpp>

#include <cstddef>
#include <iostream>

__global__ void dummy() {}

int pika_main(pika::program_options::variables_map& vm)
{
    std::size_t const iterations = vm["iterations"].as<std::size_t>();
    std::size_t const batch_size = 10;
    std::size_t const batch_iterations = iterations / batch_size;
    std::size_t const non_batch_iterations = iterations % batch_size;

    cudaStream_t cuda_stream;
    pika::cuda::experimental::check_cuda_error(cudaStreamCreate(&cuda_stream));

    // Warmup
    {
        pika::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            dummy<<<1, 1, 0, cuda_stream>>>();
            pika::cuda::experimental::check_cuda_error(
                cudaStreamSynchronize(cuda_stream));
        }
        double elapsed = timer.elapsed();
        std::cout
            << "native + synchronize (warmup):                                 "
            << elapsed << '\n';
    }

    {
        pika::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            dummy<<<1, 1, 0, cuda_stream>>>();
            pika::cuda::experimental::check_cuda_error(
                cudaStreamSynchronize(cuda_stream));
        }
        double elapsed = timer.elapsed();
        std::cout
            << "native + synchronize:                                          "
            << elapsed << '\n';
    }

    {
        pika::chrono::high_resolution_timer timer;

        for (std::size_t i = 0; i < batch_iterations; ++i)
        {
            for (std::size_t b = 0; b < batch_size; ++b)
            {
                dummy<<<1, 1, 0, cuda_stream>>>();
            }
            pika::cuda::experimental::check_cuda_error(
                cudaStreamSynchronize(cuda_stream));
        }

        for (std::size_t i = 0; i < non_batch_iterations; ++i)
        {
            dummy<<<1, 1, 0, cuda_stream>>>();
        }
        pika::cuda::experimental::check_cuda_error(
            cudaStreamSynchronize(cuda_stream));

        double elapsed = timer.elapsed();
        std::cout
            << "native + synchronize batched:                                  "
            << elapsed << '\n';
    }

    {
        pika::cuda::experimental::enable_user_polling poll("default");

        namespace ex = pika::execution::experimental;
        namespace cu = pika::cuda::experimental;

        auto const f = [](cudaStream_t cuda_stream) {
            dummy<<<1, 1, 0, cuda_stream>>>();
        };

        pika::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            cu::transform_stream(ex::just(), f, cuda_stream) | ex::sync_wait();
        }
        double elapsed = timer.elapsed();
        std::cout
            << "transform_stream:                                              "
            << elapsed << '\n';
    }

    {
        pika::cuda::experimental::enable_user_polling poll("default");

        namespace ex = pika::execution::experimental;
        namespace cu = pika::cuda::experimental;

        auto const f = [](cudaStream_t cuda_stream) {
            dummy<<<1, 1, 0, cuda_stream>>>();
        };

        pika::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i < batch_iterations; ++i)
        {
            // We have to manually unroll this loop, because the type of the
            // sender changes for each additional transform_stream call. The
            // number of unrolled calls must match batch_size above.
            cu::transform_stream(ex::just(), f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) | ex::sync_wait();
        }
        // Do the remainder one-by-one
        for (std::size_t i = 0; i < non_batch_iterations; ++i)
        {
            cu::transform_stream(ex::just(), f, cuda_stream) | ex::sync_wait();
        }
        double elapsed = timer.elapsed();
        std::cout
            << "transform_stream batched:                                      "
            << elapsed << '\n';
    }

    {
        pika::cuda::experimental::enable_user_polling poll("default");

        namespace ex = pika::execution::experimental;
        namespace cu = pika::cuda::experimental;

        auto const f = [](cudaStream_t cuda_stream) {
            dummy<<<1, 1, 0, cuda_stream>>>();
        };

        pika::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i < batch_iterations; ++i)
        {
            // We have to manually unroll this loop, because the type of the
            // sender changes for each additional transform_stream call. The
            // number of unrolled calls must match batch_size above. Here we
            // intentionally insert dummy then([]{}) calls between the
            // transform_stream calls to force synchronization between the
            // kernel launches.
            cu::transform_stream(ex::just(), f, cuda_stream) | ex::then([] {}) |
                cu::transform_stream(f, cuda_stream) | ex::then([] {}) |
                cu::transform_stream(f, cuda_stream) | ex::then([] {}) |
                cu::transform_stream(f, cuda_stream) | ex::then([] {}) |
                cu::transform_stream(f, cuda_stream) | ex::then([] {}) |
                cu::transform_stream(f, cuda_stream) | ex::then([] {}) |
                cu::transform_stream(f, cuda_stream) | ex::then([] {}) |
                cu::transform_stream(f, cuda_stream) | ex::then([] {}) |
                cu::transform_stream(f, cuda_stream) | ex::then([] {}) |
                cu::transform_stream(f, cuda_stream) | ex::sync_wait();
        }
        // Do the remainder one-by-one
        for (std::size_t i = 0; i < non_batch_iterations; ++i)
        {
            cu::transform_stream(ex::just(), f, cuda_stream) | ex::sync_wait();
        }
        double elapsed = timer.elapsed();
        std::cout
            << "transform_stream force synchronize batched:                    "
            << elapsed << '\n';
    }

    {
        pika::cuda::experimental::enable_user_polling poll("default");

        namespace ex = pika::execution::experimental;
        namespace cu = pika::cuda::experimental;

        auto const f = [](cudaStream_t cuda_stream) {
            dummy<<<1, 1, 0, cuda_stream>>>();
        };

        pika::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            cu::transform_stream(ex::just(), f, cuda_stream) |
                ex::transfer(ex::thread_pool_scheduler{}) | ex::sync_wait();
        }
        double elapsed = timer.elapsed();
        std::cout
            << "transform_stream with transfer:                                "
            << elapsed << '\n';
    }

    {
        pika::cuda::experimental::enable_user_polling poll("default");

        namespace ex = pika::execution::experimental;
        namespace cu = pika::cuda::experimental;

        auto const f = [](cudaStream_t cuda_stream) {
            dummy<<<1, 1, 0, cuda_stream>>>();
        };

        pika::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i < batch_iterations; ++i)
        {
            // We have to manually unroll this loop, because the type of the
            // sender changes for each additional transform_stream call. The
            // number of unrolled calls must match batch_size above.
            cu::transform_stream(ex::just(), f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                cu::transform_stream(f, cuda_stream) |
                ex::transfer(ex::thread_pool_scheduler{}) | ex::sync_wait();
        }
        // Do the remainder one-by-one
        for (std::size_t i = 0; i < non_batch_iterations; ++i)
        {
            cu::transform_stream(ex::just(), f, cuda_stream) |
                ex::transfer(ex::thread_pool_scheduler{}) | ex::sync_wait();
        }
        double elapsed = timer.elapsed();
        std::cout
            << "transform_stream with transfer batched:                        "
            << elapsed << '\n';
    }

    pika::cuda::experimental::check_cuda_error(cudaStreamDestroy(cuda_stream));

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    using namespace pika::program_options;

    options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");
    cmdline.add_options()("iterations",
        pika::program_options::value<std::size_t>()->default_value(1024),
        "number of iterations (default: 1024)");
    pika::local::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return pika::local::init(pika_main, argc, argv, init_args);
}
