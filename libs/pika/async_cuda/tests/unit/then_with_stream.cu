//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/cuda.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/testing.hpp>

#include <whip.hpp>

#include "algorithm_test_utils.hpp"

#include <atomic>
#include <cstddef>
#include <utility>

__global__ void dummy_kernel() {}

struct dummy
{
    static std::atomic<std::size_t> host_void_calls;
    static std::atomic<std::size_t> stream_void_calls;
    static std::atomic<std::size_t> cublas_void_calls;
    static std::atomic<std::size_t> cusolver_void_calls;
    static std::atomic<std::size_t> host_int_calls;
    static std::atomic<std::size_t> stream_int_calls;
    static std::atomic<std::size_t> cublas_int_calls;
    static std::atomic<std::size_t> cusolver_int_calls;
    static std::atomic<std::size_t> host_double_calls;
    static std::atomic<std::size_t> stream_double_calls;
    static std::atomic<std::size_t> cublas_double_calls;
    static std::atomic<std::size_t> cusolver_double_calls;

    static void reset_counts()
    {
        host_void_calls = 0;
        stream_void_calls = 0;
        cublas_void_calls = 0;
        cusolver_void_calls = 0;
        host_int_calls = 0;
        stream_int_calls = 0;
        cublas_int_calls = 0;
        cusolver_int_calls = 0;
        host_double_calls = 0;
        stream_double_calls = 0;
        cublas_double_calls = 0;
        cusolver_double_calls = 0;
    }

    void operator()() const
    {
        ++host_void_calls;
    }

    void operator()(whip::stream_t stream) const
    {
        ++stream_void_calls;
        dummy_kernel<<<1, 1, 0, stream>>>();
    }

    void operator()(cublasHandle_t) const
    {
        ++cublas_void_calls;
    }

#if !defined(PIKA_HAVE_HIP)
    // cusolverDnHandle_t and cublasHandle_t are the same so this results in a
    // redefinition with HIP.
    void operator()(cusolverDnHandle_t) const
    {
        ++cusolver_void_calls;
    }
#endif

    double operator()(int x) const
    {
        ++host_int_calls;
        return x + 1;
    }

    double operator()(int x, whip::stream_t stream) const
    {
        ++stream_int_calls;
        dummy_kernel<<<1, 1, 0, stream>>>();
        return x + 1;
    }

    double operator()(cublasHandle_t, int x) const
    {
        ++cublas_int_calls;
        return x + 1;
    }

#if !defined(PIKA_HAVE_HIP)
    double operator()(cusolverDnHandle_t, int x) const
    {
        ++cusolver_int_calls;
        return x + 1;
    }
#endif

    int operator()(double x) const
    {
        ++host_double_calls;
        return x + 1;
    }

    int operator()(double x, whip::stream_t stream) const
    {
        ++stream_double_calls;
        dummy_kernel<<<1, 1, 0, stream>>>();
        return x + 1;
    }

    int operator()(cublasHandle_t, double x) const
    {
        ++cublas_double_calls;
        return x + 1;
    }

#if !defined(PIKA_HAVE_HIP)
    int operator()(cusolverDnHandle_t, double x) const
    {
        ++cusolver_double_calls;
        return x + 1;
    }
#endif
};

std::atomic<std::size_t> dummy::host_void_calls{0};
std::atomic<std::size_t> dummy::stream_void_calls{0};
std::atomic<std::size_t> dummy::cublas_void_calls{0};
std::atomic<std::size_t> dummy::cusolver_void_calls{0};
std::atomic<std::size_t> dummy::host_int_calls{0};
std::atomic<std::size_t> dummy::stream_int_calls{0};
std::atomic<std::size_t> dummy::cublas_int_calls{0};
std::atomic<std::size_t> dummy::cusolver_int_calls{0};
std::atomic<std::size_t> dummy::host_double_calls{0};
std::atomic<std::size_t> dummy::stream_double_calls{0};
std::atomic<std::size_t> dummy::cublas_double_calls{0};
std::atomic<std::size_t> dummy::cusolver_double_calls{0};

struct dummy_stream
{
    bool& called;
    void operator()(whip::stream_t)
    {
        called = true;
    }
};

struct dummy_cublas
{
    bool& called;
    void operator()(cublasHandle_t)
    {
        called = true;
    }
};

struct dummy_cusolver
{
    bool& called;
    void operator()(cusolverDnHandle_t)
    {
        called = true;
    }
};

__global__ void increment_kernel(int* p)
{
    ++(*p);
}

struct increment
{
    int* operator()(int* p, whip::stream_t stream) const
    {
        increment_kernel<<<1, 1, 0, stream>>>(p);
        return p;
    }
};

auto non_default_constructible_params(
    custom_type_non_default_constructible& x, whip::stream_t)
{
    return std::move(x);
}
auto non_default_constructible_non_copyable_params(
    custom_type_non_default_constructible_non_copyable& x, whip::stream_t)
{
    return std::move(x);
}

int pika_main()
{
    namespace cu = ::pika::cuda::experimental;
    namespace ex = ::pika::execution::experimental;
    namespace tt = ::pika::this_thread::experimental;

    cu::cuda_pool pool{};

    cu::enable_user_polling p;

    // Only stream transform
    {
        dummy::reset_counts();
        auto s = ex::just() | ex::transfer(cu::cuda_scheduler{pool}) |
            cu::then_with_stream(dummy{});
        // NOTE: then_with_stream calls triggers the receiver on a plain
        // std::thread. We explicitly change the context back to an pika::thread.
        tt::sync_wait(ex::transfer(std::move(s), ex::thread_pool_scheduler{}));
        PIKA_TEST_EQ(dummy::host_void_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_void_calls.load(), std::size_t(1));
        PIKA_TEST_EQ(dummy::host_int_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_int_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::host_double_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_double_calls.load(), std::size_t(0));
    }

    {
        dummy::reset_counts();
        auto s = ex::just() | ex::transfer(cu::cuda_scheduler(pool)) |
            cu::then_with_stream(dummy{}) | cu::then_with_stream(dummy{}) |
            cu::then_with_stream(dummy{});
        tt::sync_wait(ex::transfer(std::move(s), ex::thread_pool_scheduler{}));
        PIKA_TEST_EQ(dummy::host_void_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_void_calls.load(), std::size_t(3));
        PIKA_TEST_EQ(dummy::host_int_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_int_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::host_double_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_double_calls.load(), std::size_t(0));
    }

    // Mixing stream transform with host scheduler
    {
        dummy::reset_counts();
        auto s = ex::just() | ex::transfer(cu::cuda_scheduler(pool)) |
            cu::then_with_stream(dummy{}) |
            ex::transfer(ex::thread_pool_scheduler{}) | ex::then(dummy{}) |
            ex::transfer(cu::cuda_scheduler(pool)) |
            cu::then_with_stream(dummy{});
        tt::sync_wait(ex::transfer(std::move(s), ex::thread_pool_scheduler{}));
        PIKA_TEST_EQ(dummy::host_void_calls.load(), std::size_t(1));
        PIKA_TEST_EQ(dummy::stream_void_calls.load(), std::size_t(2));
        PIKA_TEST_EQ(dummy::host_int_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_int_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::host_double_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_double_calls.load(), std::size_t(0));
    }

    {
        dummy::reset_counts();
        auto s = ex::schedule(ex::thread_pool_scheduler{}) | ex::then(dummy{}) |
            ex::transfer(cu::cuda_scheduler(pool)) |
            cu::then_with_stream(dummy{}) |
            ex::transfer(ex::thread_pool_scheduler{}) | ex::then(dummy{});
        tt::sync_wait(std::move(s));
        PIKA_TEST_EQ(dummy::host_void_calls.load(), std::size_t(2));
        PIKA_TEST_EQ(dummy::stream_void_calls.load(), std::size_t(1));
        PIKA_TEST_EQ(dummy::host_int_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_int_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::host_double_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_double_calls.load(), std::size_t(0));
    }

    // Only stream transform with non-void values
    {
        dummy::reset_counts();
        auto s = ex::just(1) | ex::transfer(cu::cuda_scheduler{pool}) |
            cu::then_with_stream(dummy{});
        PIKA_TEST_EQ(tt::sync_wait(ex::transfer(
                         std::move(s), ex::thread_pool_scheduler{})),
            2.0);
        PIKA_TEST_EQ(dummy::host_void_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_void_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::host_int_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_int_calls.load(), std::size_t(1));
        PIKA_TEST_EQ(dummy::host_double_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_double_calls.load(), std::size_t(0));
    }

    {
        dummy::reset_counts();
        auto s = ex::just(1) | ex::transfer(cu::cuda_scheduler{pool}) |
            cu::then_with_stream(dummy{}) | cu::then_with_stream(dummy{}) |
            cu::then_with_stream(dummy{});
        PIKA_TEST_EQ(tt::sync_wait(ex::transfer(
                         std::move(s), ex::thread_pool_scheduler{})),
            4.0);
        PIKA_TEST_EQ(dummy::host_void_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_void_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::host_int_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_int_calls.load(), std::size_t(2));
        PIKA_TEST_EQ(dummy::host_double_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_double_calls.load(), std::size_t(1));
    }

    // Non-copyable or non-default-constructible types
    {
        auto s = ex::just(custom_type_non_default_constructible{42}) |
            ex::transfer(cu::cuda_scheduler{pool}) |
            cu::then_with_stream(&non_default_constructible_params);
        PIKA_TEST_EQ(tt::sync_wait(ex::transfer(std::move(s),
                                       ex::thread_pool_scheduler{}))
                         .x,
            42);
    }

    {
        auto s =
            ex::just(custom_type_non_default_constructible_non_copyable{42}) |
            ex::transfer(cu::cuda_scheduler{pool}) |
            cu::then_with_stream(
                &non_default_constructible_non_copyable_params);
        PIKA_TEST_EQ(tt::sync_wait(ex::transfer(std::move(s),
                                       ex::thread_pool_scheduler{}))
                         .x,
            42);
    }

    // Mixing stream transform with host scheduler with non-void values
    {
        dummy::reset_counts();
        auto s = ex::just(1) | ex::transfer(cu::cuda_scheduler{pool}) |
            cu::then_with_stream(dummy{}) |
            ex::transfer(ex::thread_pool_scheduler{}) | ex::then(dummy{}) |
            ex::transfer(cu::cuda_scheduler{pool}) |
            cu::then_with_stream(dummy{});
        PIKA_TEST_EQ(tt::sync_wait(ex::transfer(
                         std::move(s), ex::thread_pool_scheduler{})),
            4.0);
        PIKA_TEST_EQ(dummy::host_void_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_void_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::host_int_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_int_calls.load(), std::size_t(2));
        PIKA_TEST_EQ(dummy::host_double_calls.load(), std::size_t(1));
        PIKA_TEST_EQ(dummy::stream_double_calls.load(), std::size_t(0));
    }

    {
        dummy::reset_counts();
        auto s = ex::just(1) | ex::transfer(ex::thread_pool_scheduler{}) |
            ex::then(dummy{}) | ex::transfer(cu::cuda_scheduler{pool}) |
            cu::then_with_stream(dummy{}) |
            ex::transfer(ex::thread_pool_scheduler{}) | ex::then(dummy{});
        PIKA_TEST_EQ(tt::sync_wait(std::move(s)), 4.0);
        PIKA_TEST_EQ(dummy::host_void_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_void_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::host_int_calls.load(), std::size_t(2));
        PIKA_TEST_EQ(dummy::stream_int_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::host_double_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_double_calls.load(), std::size_t(1));
    }

    {
        dummy::reset_counts();

        auto s = ex::transfer_just(ex::thread_pool_scheduler{}, 1) |
            ex::then(dummy{}) | ex::transfer(cu::cuda_scheduler{pool}) |
            cu::then_with_stream(dummy{}) | cu::then_with_stream(dummy{}) |
            ex::transfer(ex::thread_pool_scheduler{}) | ex::then(dummy{});
        PIKA_TEST_EQ(tt::sync_wait(std::move(s)), 5.0);
        PIKA_TEST_EQ(dummy::host_void_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_void_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::host_int_calls.load(), std::size_t(1));
        PIKA_TEST_EQ(dummy::stream_int_calls.load(), std::size_t(1));
        PIKA_TEST_EQ(dummy::host_double_calls.load(), std::size_t(1));
        PIKA_TEST_EQ(dummy::stream_double_calls.load(), std::size_t(1));
    }

    // Chaining multiple stream transforms without intermediate synchronization
    {
        whip::stream_t first_stream{};
        whip::stream_t second_stream{};
        tt::sync_wait(ex::schedule(cu::cuda_scheduler{pool}) |
            cu::then_with_stream(
                [&](whip::stream_t stream) { first_stream = stream; }) |
            cu::then_with_stream([&](whip::stream_t stream) {
                PIKA_TEST_EQ(stream, first_stream);
            }) |
            cu::then_with_stream([&](whip::stream_t stream) {
                PIKA_TEST_EQ(stream, first_stream);
            }) |
            ex::transfer(cu::cuda_scheduler{pool}) |
            cu::then_with_stream([&](whip::stream_t stream) {
                PIKA_TEST_NEQ(stream, first_stream);
                second_stream = stream;
            }) |
            cu::then_with_stream([&](whip::stream_t stream) {
                PIKA_TEST_EQ(stream, second_stream);
            }) |
            cu::then_with_stream([&](whip::stream_t stream) {
                PIKA_TEST_EQ(stream, second_stream);
            }));
    }

    {
        using type = int;
        type p_h = 0;

        type* p;
        whip::malloc(&p, sizeof(type));

        auto s = ex::just(p, &p_h, sizeof(type), whip::memcpy_host_to_device) |
            ex::transfer(cu::cuda_scheduler{pool}) |
            cu::then_with_stream(whip::memcpy_async) |
            ex::transfer(ex::thread_pool_scheduler{}) |
            ex::then([p] { return p; }) |
            ex::transfer(cu::cuda_scheduler{pool}) |
            cu::then_with_stream(increment{}) |
            cu::then_with_stream(increment{}) |
            cu::then_with_stream(increment{});
        tt::sync_wait(
            ex::when_all(ex::just(&p_h), std::move(s), ex::just(sizeof(type)),
                ex::just(whip::memcpy_device_to_host)) |
            ex::transfer(cu::cuda_scheduler{pool}) |
            cu::then_with_stream(whip::memcpy_async) |
            ex::transfer(ex::thread_pool_scheduler{}) |
            ex::then([&p_h] { PIKA_TEST_EQ(p_h, 3); }) |
            ex::transfer(ex::thread_pool_scheduler{}));

        whip::free(p);
    }

    // cuBLAS and cuSOLVER
    {
        dummy::reset_counts();
        auto s = ex::just(1) | ex::transfer(ex::thread_pool_scheduler{}) |
            ex::then(dummy{}) | ex::transfer(cu::cuda_scheduler{pool}) |
            cu::then_with_stream(dummy{}) |
            cu::then_with_cublas(dummy{}, CUBLAS_POINTER_MODE_HOST) |
            cu::then_with_cusolver(dummy{}) |
            ex::transfer(ex::thread_pool_scheduler{}) | ex::then(dummy{});
        PIKA_TEST_EQ(tt::sync_wait(std::move(s)), 6);
        PIKA_TEST_EQ(dummy::host_void_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_void_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::cublas_void_calls.load(), std::size_t(0));
#if !defined(PIKA_HAVE_HIP)
        PIKA_TEST_EQ(dummy::cusolver_void_calls.load(), std::size_t(0));
#endif
        PIKA_TEST_EQ(dummy::host_int_calls.load(), std::size_t(2));
        PIKA_TEST_EQ(dummy::stream_int_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::cublas_int_calls.load(), std::size_t(1));
#if !defined(PIKA_HAVE_HIP)
        PIKA_TEST_EQ(dummy::cusolver_int_calls.load(), std::size_t(0));
#endif
        PIKA_TEST_EQ(dummy::host_double_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_double_calls.load(), std::size_t(1));
#if defined(PIKA_HAVE_HIP)
        // rocblas_handle and rocsolver_handle being the same, the call to
        // then_with_cusolver results in a increment of the cublas overload.
        PIKA_TEST_EQ(dummy::cublas_double_calls.load(), std::size_t(1));
#else
        PIKA_TEST_EQ(dummy::cublas_double_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::cusolver_double_calls.load(), std::size_t(1));
#endif
    }

    {
        dummy::reset_counts();
        auto s = ex::just(1) | ex::transfer(ex::thread_pool_scheduler{}) |
            ex::then(dummy{}) | ex::transfer(cu::cuda_scheduler{pool}) |
            cu::then_with_stream(dummy{}) | cu::then_on_host(dummy{}) |
            cu::then_with_cublas(dummy{}, CUBLAS_POINTER_MODE_HOST) |
            cu::then_with_cusolver(dummy{}) |
            ex::transfer(ex::thread_pool_scheduler{}) | ex::then(dummy{});
        PIKA_TEST_EQ(tt::sync_wait(std::move(s)), 7.0);
        PIKA_TEST_EQ(dummy::host_void_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::stream_void_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::cublas_void_calls.load(), std::size_t(0));
#if !defined(PIKA_HAVE_HIP)
        PIKA_TEST_EQ(dummy::cusolver_void_calls.load(), std::size_t(0));
#endif
        PIKA_TEST_EQ(dummy::host_int_calls.load(), std::size_t(2));
        PIKA_TEST_EQ(dummy::stream_int_calls.load(), std::size_t(0));
#if defined(PIKA_HAVE_HIP)
        // rocblas_handle and rocsolver_handle being the same, the call to
        // then_with_cusolver results in a increment of the cublas overload.
        PIKA_TEST_EQ(dummy::cublas_int_calls.load(), std::size_t(1));
#else
        PIKA_TEST_EQ(dummy::cublas_int_calls.load(), std::size_t(0));
        PIKA_TEST_EQ(dummy::cusolver_int_calls.load(), std::size_t(1));
#endif
        PIKA_TEST_EQ(dummy::host_double_calls.load(), std::size_t(1));
        PIKA_TEST_EQ(dummy::stream_double_calls.load(), std::size_t(1));
        PIKA_TEST_EQ(dummy::cublas_double_calls.load(), std::size_t(1));
#if !defined(PIKA_HAVE_HIP)
        PIKA_TEST_EQ(dummy::cusolver_double_calls.load(), std::size_t(0));
#endif
    }

    return pika::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return 0;
}
