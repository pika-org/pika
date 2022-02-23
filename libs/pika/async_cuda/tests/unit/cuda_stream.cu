//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/async_cuda/cuda_exception.hpp>
#include <pika/async_cuda/cuda_stream.hpp>
#include <pika/testing.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace cu = pika::cuda::experimental;

__global__ void kernel(int* p, int i)
{
    p[i] = i * 2;
}

int main()
{
    {
        // Default constructed cuda_stream uses device 0 and default priority
        cu::cuda_stream stream{};

        PIKA_TEST_EQ(stream.get_device(), 0);
        PIKA_TEST_EQ(
            stream.get_priority(), pika::threads::thread_priority::default_);

        PIKA_TEST_NEQ(stream.get(), cudaStream_t{});

        cu::cuda_stream stream2{std::move(stream)};

        PIKA_TEST_EQ(stream.get(), cudaStream_t{});
        PIKA_TEST_NEQ(stream2.get(), cudaStream_t{});
    }

    {
        // We can't really test setting the device properly unless we have
        // multiple devices available, but we test the constructor anyway. The
        // other behaviour should be the same as for a default constructed
        // cuda_stream.
        cu::cuda_stream stream{0};

        PIKA_TEST_EQ(stream.get_device(), 0);
        PIKA_TEST_EQ(
            stream.get_priority(), pika::threads::thread_priority::default_);

        PIKA_TEST_NEQ(stream.get(), cudaStream_t{});

        cu::cuda_stream stream2{std::move(stream)};

        PIKA_TEST_EQ(stream.get(), cudaStream_t{});
        PIKA_TEST_NEQ(stream2.get(), cudaStream_t{});
    }

    {
        // We should also be able to set the priority.
        cu::cuda_stream stream{0, pika::threads::thread_priority::normal};

        PIKA_TEST_EQ(stream.get_device(), 0);
        PIKA_TEST_EQ(
            stream.get_priority(), pika::threads::thread_priority::normal);

        PIKA_TEST_NEQ(stream.get(), cudaStream_t{});

        cu::cuda_stream stream2{std::move(stream)};

        PIKA_TEST_EQ(stream.get(), cudaStream_t{});
        PIKA_TEST_NEQ(stream2.get(), cudaStream_t{});
        PIKA_TEST_EQ(
            stream.get_priority(), pika::threads::thread_priority::default_);
        PIKA_TEST_EQ(
            stream2.get_priority(), pika::threads::thread_priority::normal);

        cu::cuda_stream stream3{stream};
        cu::cuda_stream stream4{stream2};

        PIKA_TEST_EQ(stream3.get(), cudaStream_t{});
        PIKA_TEST_NEQ(stream4.get(), cudaStream_t{});
        PIKA_TEST_NEQ(stream4.get(), stream2.get());
        PIKA_TEST_EQ(
            stream3.get_priority(), pika::threads::thread_priority::default_);
        PIKA_TEST_EQ(
            stream4.get_priority(), pika::threads::thread_priority::normal);
    }

    {
        cu::cuda_stream stream{0, pika::threads::thread_priority::high};

        PIKA_TEST_EQ(stream.get_device(), 0);
        PIKA_TEST_EQ(
            stream.get_priority(), pika::threads::thread_priority::high);

        PIKA_TEST_NEQ(stream.get(), cudaStream_t{});

        cu::cuda_stream stream2{std::move(stream)};

        PIKA_TEST_EQ(stream.get(), cudaStream_t{});
        PIKA_TEST_NEQ(stream2.get(), cudaStream_t{});
        PIKA_TEST_EQ(
            stream.get_priority(), pika::threads::thread_priority::default_);
        PIKA_TEST_EQ(
            stream2.get_priority(), pika::threads::thread_priority::high);

        cu::cuda_stream stream3{stream};
        cu::cuda_stream stream4{stream2};

        PIKA_TEST_EQ(stream3.get(), cudaStream_t{});
        PIKA_TEST_NEQ(stream4.get(), cudaStream_t{});
        PIKA_TEST_NEQ(stream4.get(), stream2.get());
        PIKA_TEST_EQ(
            stream3.get_priority(), pika::threads::thread_priority::default_);
        PIKA_TEST_EQ(
            stream4.get_priority(), pika::threads::thread_priority::high);
    }

    {
        // Equality is based on the underlying stream.
        cu::cuda_stream stream1{0, pika::threads::thread_priority::normal};
        cu::cuda_stream stream2{0, pika::threads::thread_priority::normal};
        cu::cuda_stream stream3{0, pika::threads::thread_priority::high};

        PIKA_TEST_NEQ(stream1, stream2);
        PIKA_TEST_NEQ(stream1, stream3);

        cu::cuda_stream stream4{std::move(stream1)};
        cu::cuda_stream stream5{std::move(stream2)};
        cu::cuda_stream stream6{std::move(stream3)};

        PIKA_TEST_EQ(stream1, stream2);
        PIKA_TEST_EQ(stream1, stream3);
        PIKA_TEST_NEQ(stream4, stream5);
        PIKA_TEST_NEQ(stream4, stream6);
    }

    {
        // We can schedule work with the underlying stream in a cuda_stream.
        std::vector<cu::cuda_stream> streams;
        streams.emplace_back();
        streams.emplace_back();
        streams.emplace_back(0);
        streams.emplace_back(0, pika::threads::thread_priority::normal);
        streams.emplace_back(0, pika::threads::thread_priority::high);
        // The first stream should stay usable after this
        streams.push_back(std::move(streams[0]));

        int* p;
        cu::check_cuda_error(cudaMalloc(&p, sizeof(int) * streams.size()));

        for (std::size_t i = 0; i < streams.size(); ++i)
        {
            kernel<<<1, 1, 0, streams[i].get()>>>(p, i);
            cu::check_cuda_error(cudaGetLastError());
        }

        cu::check_cuda_error(cudaDeviceSynchronize());
        std::vector<int> s(streams.size(), 0);

        cu::check_cuda_error(cudaMemcpy(
            s.data(), p, sizeof(int) * streams.size(), cudaMemcpyDeviceToHost));
        cu::check_cuda_error(cudaFree(p));

        for (int i = 0; i < static_cast<int>(streams.size()); ++i)
        {
            PIKA_TEST_EQ(s[i], i * 2);
        }
    }
}
