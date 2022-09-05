//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/cuda.hpp>
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
        // A pool with only one stream always gives the same stream
        cu::cuda_pool pool{0, 1, 1};

        auto& stream1 = pool.get_next_stream();
        auto& stream2 = pool.get_next_stream();
        auto& stream3 = pool.get_next_stream();

        PIKA_TEST_EQ(stream1, stream2);
        PIKA_TEST_EQ(stream1, stream3);

        auto& hpstream1 =
            pool.get_next_stream(pika::execution::thread_priority::high);
        auto& hpstream2 =
            pool.get_next_stream(pika::execution::thread_priority::high);
        auto& hpstream3 =
            pool.get_next_stream(pika::execution::thread_priority::high);

        PIKA_TEST_EQ(hpstream1, hpstream2);
        PIKA_TEST_EQ(hpstream1, hpstream3);
    }

    {
        // A pool with multiple streams cycles through the streams
        cu::cuda_pool pool{0, 3, 2};

        auto& stream1 = pool.get_next_stream();
        auto& stream2 = pool.get_next_stream();
        auto& stream3 = pool.get_next_stream();
        auto& stream4 = pool.get_next_stream();
        auto& stream5 = pool.get_next_stream();
        auto& stream6 = pool.get_next_stream();

        PIKA_TEST_EQ(stream1, stream4);
        PIKA_TEST_EQ(stream2, stream5);
        PIKA_TEST_EQ(stream3, stream6);
        PIKA_TEST_NEQ(stream1, stream2);
        PIKA_TEST_NEQ(stream1, stream3);
        PIKA_TEST_NEQ(stream2, stream3);

        auto& hpstream1 =
            pool.get_next_stream(pika::execution::thread_priority::high);
        auto& hpstream2 =
            pool.get_next_stream(pika::execution::thread_priority::high);
        auto& hpstream3 =
            pool.get_next_stream(pika::execution::thread_priority::high);
        auto& hpstream4 =
            pool.get_next_stream(pika::execution::thread_priority::high);

        PIKA_TEST_EQ(hpstream1, hpstream3);
        PIKA_TEST_EQ(hpstream2, hpstream4);
        PIKA_TEST_NEQ(hpstream1, hpstream2);
    }

    // Flags can be set on streams in the pool
    {
        cu::cuda_pool pool{0, 3, 2};

        unsigned int expected_flags = 0;
        unsigned int flags = 0;
        cu::cuda_stream const& stream = pool.get_next_stream();
        whip::stream_get_flags(stream.get(), &flags);
        PIKA_TEST_EQ(stream.get_flags(), expected_flags);
        PIKA_TEST_EQ(flags, expected_flags);
    }

    {
        unsigned int expected_flags = 0;
        cu::cuda_pool pool{0, 3, 2, expected_flags};

        unsigned int flags = 0;
        cu::cuda_stream const& stream = pool.get_next_stream();
        whip::stream_get_flags(stream.get(), &flags);
        PIKA_TEST_EQ(stream.get_flags(), expected_flags);
        PIKA_TEST_EQ(flags, expected_flags);
    }

    {
        unsigned int expected_flags = whip::stream_non_blocking;
        cu::cuda_pool pool{0, 3, 2, expected_flags};

        unsigned int flags = 0;
        cu::cuda_stream const& stream = pool.get_next_stream();
        whip::stream_get_flags(stream.get(), &flags);
        PIKA_TEST_EQ(stream.get_flags(), expected_flags);
        PIKA_TEST_EQ(flags, expected_flags);
    }

    {
        // A pool is reference counted
        cu::cuda_pool pool{};
        PIKA_TEST(pool.valid());
        PIKA_TEST(bool(pool));

        cu::cuda_pool pool2{pool};
        PIKA_TEST(pool2.valid());
        PIKA_TEST(bool(pool2));
        PIKA_TEST_EQ(pool, pool2);

        cu::cuda_pool pool3 = pool;
        PIKA_TEST(pool3.valid());
        PIKA_TEST(bool(pool3));
        PIKA_TEST_EQ(pool, pool3);

        cu::cuda_pool pool4{std::move(pool)};
        PIKA_TEST(!pool.valid());
        PIKA_TEST(!bool(pool));
        PIKA_TEST(pool4.valid());
        PIKA_TEST(bool(pool4));
        PIKA_TEST_NEQ(pool, pool4);

        cu::cuda_pool pool5{std::move(pool4)};
        PIKA_TEST(!pool4.valid());
        PIKA_TEST(!bool(pool4));
        PIKA_TEST(pool5.valid());
        PIKA_TEST(bool(pool5));
        PIKA_TEST_NEQ(pool4, pool5);
    }

    {
        // A pool can be used to schedule work
        int const n = 1000;
        int* p;
        whip::malloc(&p, sizeof(int) * n);

        cu::cuda_pool pool{};

        for (std::size_t i = 0; i < n; ++i)
        {
            kernel<<<1, 1, 0, pool.get_next_stream().get()>>>(p, i);
            whip::check_last_error();
        }

        whip::device_synchronize();
        std::vector<int> s(n, 0);

        whip::memcpy(s.data(), p, sizeof(int) * n, whip::memcpy_device_to_host);
        whip::free(p);

        for (int i = 0; i < n; ++i)
        {
            PIKA_TEST_EQ(s[i], i * 2);
        }
    }
}
