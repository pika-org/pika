//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/cuda.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>

#include <fmt/printf.h>
#include <whip.hpp>

#include <cstddef>
#include <tuple>
#include <utility>

#if defined(PIKA_HAVE_CUDA)
# include <cublas_v2.h>
using blas_handle_t = cublasHandle_t;
auto* blas_gemm = &cublasDgemm;
auto blas_pointer_mode = CUBLAS_POINTER_MODE_HOST;
auto blas_op_n = CUBLAS_OP_N;
#elif defined(PIKA_HAVE_HIP)
# include <rocblas/rocblas.h>
using blas_handle_t = rocblas_handle;
# define CUBLAS_POINTER_MODE_HOST rocblas_pointer_mode_host
auto* blas_gemm = &rocblas_dgemm;
auto blas_pointer_mode = rocblas_pointer_mode_host;
auto blas_op_n = rocblas_operation_none;
#endif

// Owning wrapper for GPU-allocated memory.
class gpu_data
{
    double* p{nullptr};
    std::size_t n{0};

public:
    // Note that blocking functions such as cudaMalloc will block the underlying operating system
    // thread instead of yielding the pika task. Consider using e.g. a pool of GPU memory to avoid
    // blocking the thread for too long.
    gpu_data(std::size_t n)
      : n(n)
    {
        whip::malloc(&p, sizeof(double) * n);
    }
    gpu_data(gpu_data&& other) noexcept
      : p(std::exchange(other.p, nullptr))
      , n(std::exchange(other.n, 0))
    {
    }
    gpu_data& operator=(gpu_data&& other) noexcept
    {
        p = std::exchange(other.p, nullptr);
        n = std::exchange(other.n, 0);
        return *this;
    }
    gpu_data(gpu_data const&) = delete;
    gpu_data& operator=(gpu_data const&) = delete;
    ~gpu_data() { whip::free(p); }

    std::size_t size() const { return n; }
    double* get() const { return p; }
};

__global__ void init(double* a, double* b, double* c, std::size_t n)
{
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 3.0;
    }
}

int main(int argc, char* argv[])
{
    namespace cu = pika::cuda::experimental;
    namespace ex = pika::execution::experimental;
    namespace tt = pika::this_thread::experimental;

    pika::start(argc, argv);
    ex::thread_pool_scheduler cpu_sched{};
    cu::cuda_pool pool{};
    cu::cuda_scheduler cuda_sched{pool};

    {
        cu::enable_user_polling p{};

        constexpr std::size_t n = 2048;
        gpu_data a{n * n};
        gpu_data b{n * n};
        gpu_data c{n * n};
        double alpha = 1.0;
        double beta = 1.0;

        auto s = ex::just(std::move(a), std::move(b), std::move(c)) | ex::continues_on(cuda_sched) |
            cu::then_with_stream([](auto& a, auto& b, auto& c, whip::stream_t stream) {
                init<<<n * n / 256, 256, 0, stream>>>(a.get(), b.get(), c.get(), n * n);
                return std::make_tuple(std::move(a), std::move(b), std::move(c));
            }) |
            ex::unpack() |
            // a, b, and c will be kept alive by the then_with_cublas operation state at least until
            // the GPU kernels complete.  Values sent by the predecessor sender are passed as the
            // last arguments after the handle.
            cu::then_with_cublas(
                [&](blas_handle_t handle, auto& a, auto& b, auto& c) {
                    blas_gemm(handle, blas_op_n, blas_op_n, n, n, n, &alpha, a.get(), n, b.get(), n,
                        &beta, c.get(), n);
                },
                blas_pointer_mode);
        tt::sync_wait(std::move(s));
    }

    pika::finalize();
    pika::stop();

    return 0;
}
