//  Copyright (c) 2017 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/chrono.hpp>
#include <pika/cuda.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>

#include <cstddef>
#include <iostream>

__global__ void dummy() {}

int pika_main(pika::program_options::variables_map& vm)
{
    namespace ex = pika::execution::experimental;
    namespace cu = pika::cuda::experimental;

    std::size_t const iterations = vm["iterations"].as<std::size_t>();
    std::size_t const batch_size = 10;
    std::size_t const batch_iterations = iterations / batch_size;
    std::size_t const non_batch_iterations = iterations % batch_size;

    cu::cuda_pool pool{};
    cu::cuda_scheduler sched{pool};
    cudaStream_t cuda_stream = pool.get_next_stream().get();

    // Warmup
    {
        pika::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            dummy<<<1, 1, 0, cuda_stream>>>();
            cu::check_cuda_error(cudaStreamSynchronize(cuda_stream));
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
            cu::check_cuda_error(cudaStreamSynchronize(cuda_stream));
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
            cu::check_cuda_error(cudaStreamSynchronize(cuda_stream));
        }

        for (std::size_t i = 0; i < non_batch_iterations; ++i)
        {
            dummy<<<1, 1, 0, cuda_stream>>>();
        }
        cu::check_cuda_error(cudaStreamSynchronize(cuda_stream));

        double elapsed = timer.elapsed();
        std::cout
            << "native + synchronize batched:                                  "
            << elapsed << '\n';
    }

    {
        cu::enable_user_polling poll("default");

        auto const f = [](cudaStream_t cuda_stream) {
            dummy<<<1, 1, 0, cuda_stream>>>();
        };

        pika::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            ex::schedule(sched) | cu::then_with_stream(f) | ex::sync_wait();
        }
        double elapsed = timer.elapsed();
        std::cout
            << "then_with_stream:                                              "
            << elapsed << '\n';
    }

    {
        cu::enable_user_polling poll("default");

        auto const f = [](cudaStream_t cuda_stream) {
            dummy<<<1, 1, 0, cuda_stream>>>();
        };

        pika::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i < batch_iterations; ++i)
        {
            // We have to manually unroll this loop, because the type of the
            // sender changes for each additional then_with_stream call. The
            // number of unrolled calls must match batch_size above.
            ex::schedule(sched) | cu::then_with_stream(f) |
                cu::then_with_stream(f) | cu::then_with_stream(f) |
                cu::then_with_stream(f) | cu::then_with_stream(f) |
                cu::then_with_stream(f) | cu::then_with_stream(f) |
                cu::then_with_stream(f) | cu::then_with_stream(f) |
                cu::then_with_stream(f) | ex::sync_wait();
        }
        // Do the remainder one-by-one
        for (std::size_t i = 0; i < non_batch_iterations; ++i)
        {
            ex::schedule(sched) | cu::then_with_stream(f) | ex::sync_wait();
        }
        double elapsed = timer.elapsed();
        std::cout
            << "then_with_stream batched:                                      "
            << elapsed << '\n';
    }

    {
        cu::enable_user_polling poll("default");

        auto const f = [](cudaStream_t cuda_stream) {
            dummy<<<1, 1, 0, cuda_stream>>>();
        };

        pika::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i < batch_iterations; ++i)
        {
            // We have to manually unroll this loop, because the type of the
            // sender changes for each additional then_with_stream call. The
            // number of unrolled calls must match batch_size above. Here we
            // intentionally insert dummy then([]{}) calls between the
            // then_with_stream calls to force synchronization between the
            // kernel launches.
            ex::schedule(sched) | cu::then_with_stream(f) |
                ex::transfer(ex::thread_pool_scheduler{}) | ex::then([] {}) |
                ex::transfer(sched) | cu::then_with_stream(f) |
                ex::transfer(ex::thread_pool_scheduler{}) | ex::then([] {}) |
                ex::transfer(sched) | cu::then_with_stream(f) |
                ex::transfer(ex::thread_pool_scheduler{}) | ex::then([] {}) |
                ex::transfer(sched) | cu::then_with_stream(f) |
                ex::transfer(ex::thread_pool_scheduler{}) | ex::then([] {}) |
                ex::transfer(sched) | cu::then_with_stream(f) |
                ex::transfer(ex::thread_pool_scheduler{}) | ex::then([] {}) |
                ex::transfer(sched) | cu::then_with_stream(f) |
                ex::transfer(ex::thread_pool_scheduler{}) | ex::then([] {}) |
                ex::transfer(sched) | cu::then_with_stream(f) |
                ex::transfer(ex::thread_pool_scheduler{}) | ex::then([] {}) |
                ex::transfer(sched) | cu::then_with_stream(f) |
                ex::transfer(ex::thread_pool_scheduler{}) | ex::then([] {}) |
                ex::transfer(sched) | cu::then_with_stream(f) |
                ex::transfer(ex::thread_pool_scheduler{}) | ex::then([] {}) |
                ex::transfer(sched) | cu::then_with_stream(f) | ex::sync_wait();
        }
        // Do the remainder one-by-one
        for (std::size_t i = 0; i < non_batch_iterations; ++i)
        {
            ex::schedule(sched) | cu::then_with_stream(f) | ex::sync_wait();
        }
        double elapsed = timer.elapsed();
        std::cout
            << "then_with_stream force synchronize batched:                    "
            << elapsed << '\n';
    }

    {
        cu::enable_user_polling poll("default");

        auto const f = [](cudaStream_t cuda_stream) {
            dummy<<<1, 1, 0, cuda_stream>>>();
        };

        pika::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i != iterations; ++i)
        {
            ex::schedule(sched) | cu::then_with_stream(f) |
                ex::transfer(ex::thread_pool_scheduler{}) | ex::sync_wait();
        }
        double elapsed = timer.elapsed();
        std::cout
            << "then_with_stream with transfer:                                "
            << elapsed << '\n';
    }

    {
        cu::enable_user_polling poll("default");

        auto const f = [](cudaStream_t cuda_stream) {
            dummy<<<1, 1, 0, cuda_stream>>>();
        };

        pika::chrono::high_resolution_timer timer;
        for (std::size_t i = 0; i < batch_iterations; ++i)
        {
            // We have to manually unroll this loop, because the type of the
            // sender changes for each additional then_with_stream call. The
            // number of unrolled calls must match batch_size above.
            ex::schedule(sched) | cu::then_with_stream(f) |
                cu::then_with_stream(f) | cu::then_with_stream(f) |
                cu::then_with_stream(f) | cu::then_with_stream(f) |
                cu::then_with_stream(f) | cu::then_with_stream(f) |
                cu::then_with_stream(f) | cu::then_with_stream(f) |
                cu::then_with_stream(f) |
                ex::transfer(ex::thread_pool_scheduler{}) | ex::sync_wait();
        }
        // Do the remainder one-by-one
        for (std::size_t i = 0; i < non_batch_iterations; ++i)
        {
            ex::schedule(sched) | cu::then_with_stream(f) |
                ex::transfer(ex::thread_pool_scheduler{}) | ex::sync_wait();
        }
        double elapsed = timer.elapsed();
        std::cout
            << "then_with_stream with transfer batched:                        "
            << elapsed << '\n';
    }

    return pika::finalize();
}

int main(int argc, char* argv[])
{
    using namespace pika::program_options;

    options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");
    cmdline.add_options()("iterations",
        pika::program_options::value<std::size_t>()->default_value(1024),
        "number of iterations (default: 1024)");
    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return pika::init(pika_main, argc, argv, init_args);
}
