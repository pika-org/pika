//  Copyright (c) 2017 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/chrono.hpp>
#include <pika/cuda.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/testing.hpp>

#include <whip.hpp>

#include <cstddef>
#include <utility>
#include <vector>

int pika_main()
{
    namespace cu = pika::cuda::experimental;
    namespace ex = pika::execution::experimental;
    namespace tt = pika::this_thread::experimental;

    cu::enable_user_polling poll("default");
    cu::cuda_pool pool{};
    cu::cuda_scheduler sched{pool};

    using element_type = int;

    auto malloc = [](void* p, element_type n, whip::stream_t stream) {
#if PIKA_CUDA_VERSION >= 1102
        whip::malloc_async(&p, sizeof(element_type) * n, stream);
#else
        // This is not a good idea in real code, but is good enough for the
        // purposes of this test.
        whip::malloc(&p, sizeof(element_type) * n);
        PIKA_UNUSED(stream);
#endif
        return p;
    };
    auto f = [] PIKA_HOST_DEVICE(element_type i, void* p) { static_cast<element_type*>(p)[i] = i; };
    auto free = [](void* p, whip::stream_t stream) {
#if PIKA_CUDA_VERSION >= 1102
        whip::free_async(p, stream);
#else
        // This is not a good idea in real code, but is good enough for the
        // purposes of this test.
        whip::free(p);
        PIKA_UNUSED(stream);
#endif
    };

    // Integral shape
    for (element_type n : {1, 42, 10007})
    {
        std::vector<element_type> host_vector(n, 0);
        element_type* device_ptr = nullptr;

        // We capture the host vector and the size into the lambda. The proper
        // way would be to use a when_all, but currently using when_all would
        // mean additional synchronization.
        auto memcpy = [&](void* p, whip::stream_t stream) {
            whip::memcpy_async(host_vector.data(), p, sizeof(element_type) * n,
                whip::memcpy_device_to_host, stream);
            return p;
        };

        auto s = ex::transfer_just(sched, device_ptr, n) | cu::then_with_stream(malloc) |
            ex::bulk(n, f) | cu::then_with_stream(memcpy) | cu::then_with_stream(free);
        tt::sync_wait(std::move(s));

        for (element_type i = 0; i < n; ++i) { PIKA_TEST_EQ(host_vector[i], i); }
    }

    // The reference implementation does not support ranges as the shape
#if !defined(PIKA_HAVE_STDEXEC)
    // Range
    for (element_type n : {1, 42, 10007})
    {
        using element_type = int;

        std::vector<element_type> host_vector(n, 0);
        element_type* device_ptr = nullptr;

        // We capture the host vector the size into the lambda. The proper way
        // would be to use a when_all, but currently using when_all would mean
        // additional synchronization.
        auto memcpy = [&](void* p, whip::stream_t stream) {
            whip::memcpy_async(host_vector.data(), p, sizeof(element_type) * n,
                whip::memcpy_device_to_host, stream);
            return p;
        };

        auto s = ex::transfer_just(sched, device_ptr, n) | cu::then_with_stream(malloc) |
            ex::bulk(pika::util::detail::make_counting_shape(n), f) | cu::then_with_stream(memcpy) |
            cu::then_with_stream(free);
        tt::sync_wait(std::move(s));

        for (element_type i = 0; i < n; ++i) { PIKA_TEST_EQ(host_vector[i], i); }
    }
#endif

    return pika::finalize();
}

int main(int argc, char* argv[]) { return pika::init(pika_main, argc, argv); }
