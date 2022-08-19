//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/cuda.hpp>
#include <pika/testing.hpp>

#include <utility>

#if defined(PIKA_HAVE_HIP)
#define cublasSasum rocblas_sasum
#endif

namespace cu = pika::cuda::experimental;

__global__ void kernel(float* p)
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    p[i] = i;
}

int main()
{
    cu::cuda_stream stream;

    {
        // Default constructed cublas_handle uses device 0 and default priority
        cu::cublas_handle handle{};

        PIKA_TEST_EQ(handle.get_device(), 0);
        PIKA_TEST_EQ(handle.get_stream(), cudaStream_t{0});

        PIKA_TEST_NEQ(handle.get(), cublasHandle_t{});

        cu::cublas_handle handle2{std::move(handle)};

        PIKA_TEST_EQ(handle.get(), cublasHandle_t{});
        PIKA_TEST_NEQ(handle2.get(), cublasHandle_t{});

        cu::cublas_handle handle3{handle};
        cu::cublas_handle handle4{handle2};

        PIKA_TEST_EQ(handle3.get(), cublasHandle_t{});
        PIKA_TEST_NEQ(handle4.get(), cublasHandle_t{});
        PIKA_TEST_NEQ(handle4.get(), handle2.get());
    }

    {
        // Equality is based on the underlying handle.
        cu::cublas_handle handle1{stream};
        cu::cublas_handle handle2{stream};
        cu::cublas_handle handle3{stream};

        PIKA_TEST_NEQ(handle1.get_stream(), cudaStream_t{0});
        PIKA_TEST_NEQ(handle2.get_stream(), cudaStream_t{0});
        PIKA_TEST_NEQ(handle3.get_stream(), cudaStream_t{0});
        PIKA_TEST_EQ(handle1.get_stream(), stream.get());
        PIKA_TEST_EQ(handle2.get_stream(), stream.get());
        PIKA_TEST_EQ(handle3.get_stream(), stream.get());
        PIKA_TEST_NEQ(handle1, handle2);
        PIKA_TEST_NEQ(handle1, handle3);

        cu::cublas_handle handle4{std::move(handle1)};
        cu::cublas_handle handle5{std::move(handle2)};
        cu::cublas_handle handle6{std::move(handle3)};

        PIKA_TEST_EQ(handle1, handle2);
        PIKA_TEST_EQ(handle1, handle3);
        PIKA_TEST_NEQ(handle4, handle5);
        PIKA_TEST_NEQ(handle4, handle6);
    }

    {
        // Equality is based on the underlying handle.
        cu::cublas_handle handle1{};
        cu::cublas_handle handle2{};
        cu::cublas_handle handle3{};

        handle1.set_stream(stream);
        handle2.set_stream(stream);
        handle3.set_stream(stream);

        PIKA_TEST_NEQ(handle1.get_stream(), cudaStream_t{0});
        PIKA_TEST_NEQ(handle2.get_stream(), cudaStream_t{0});
        PIKA_TEST_NEQ(handle3.get_stream(), cudaStream_t{0});
        PIKA_TEST_EQ(handle1.get_stream(), stream.get());
        PIKA_TEST_EQ(handle2.get_stream(), stream.get());
        PIKA_TEST_EQ(handle3.get_stream(), stream.get());
        PIKA_TEST_NEQ(handle1, handle2);
        PIKA_TEST_NEQ(handle1, handle3);

        cu::cublas_handle handle4{std::move(handle1)};
        cu::cublas_handle handle5{std::move(handle2)};
        cu::cublas_handle handle6{std::move(handle3)};

        PIKA_TEST_EQ(handle1, handle2);
        PIKA_TEST_EQ(handle1, handle3);
        PIKA_TEST_NEQ(handle4, handle5);
        PIKA_TEST_NEQ(handle4, handle6);
    }

    {
        // We can schedule work with the underlying handle in a cublas_handle.
        cu::cublas_handle handle{stream};

        int const n = 100;
        float* p;
        cu::check_cuda_error(cudaMalloc(&p, sizeof(float) * n));

        kernel<<<n, 1, 0, handle.get_stream()>>>(p);
        cu::check_cuda_error(cudaGetLastError());
        cu::check_cuda_error(cudaDeviceSynchronize());
        float r;
        handle.set_pointer_mode(CUBLAS_POINTER_MODE_HOST);
        cu::check_cublas_error(cublasSasum(handle.get(), n, p, 1, &r));
        cu::check_cuda_error(cudaDeviceSynchronize());

        cu::check_cuda_error(cudaFree(p));

        PIKA_TEST_EQ(r, (n * (n - 1) / 2));
    }

    {
        // A moved-from handle is invalid and will give an error if used
        cu::cublas_handle handle{stream};
        cu::cublas_handle handle2{std::move(handle)};

        try
        {
            cu::check_cublas_error(
                cublasSetPointerMode(handle.get(), CUBLAS_POINTER_MODE_HOST));
            PIKA_TEST(false);
        }
        catch (cu::cublas_exception const& e)
        {
            PIKA_TEST_EQ(
                e.get_cublas_errorcode(), CUBLAS_STATUS_NOT_INITIALIZED);
        }
        catch (...)
        {
            PIKA_TEST(false);
        }
    }
}
