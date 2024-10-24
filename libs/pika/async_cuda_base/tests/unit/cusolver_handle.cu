//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/cuda.hpp>
#include <pika/testing.hpp>

#include <whip.hpp>

#include <iostream>
#include <utility>

#if defined(PIKA_HAVE_HIP)
# define CUBLAS_OP_N rocblas_operation_none
# define cusolverDnDgetrs rocsolver_dgetrs
# define cusolverDnDgetrf rocsolver_dgetrf
#endif

namespace cu = pika::cuda::experimental;

__global__ void kernel(float* p)
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    p[i] = i;
}

void print_matrix(int m, int n, double const* A, int lda, char const* name)
{
    for (int row = 0; row < m; row++)
    {
        for (int col = 0; col < n; col++)
        {
            std::cout << name << "(" << row << "," << col << ") = " << A[row + col * lda] << "\n";
        }
    }
}

int main()
{
    cu::cuda_stream stream;

    {
        // Default constructed cusolver_handle uses device 0 and default priority
        cu::cusolver_handle handle{};

        PIKA_TEST_EQ(handle.get_device(), 0);
        PIKA_TEST_EQ(handle.get_stream(), whip::stream_t{0});

        PIKA_TEST_NEQ(handle.get(), cusolverDnHandle_t{});

        cu::cusolver_handle handle2{std::move(handle)};

        PIKA_TEST_EQ(handle.get(), cusolverDnHandle_t{});
        PIKA_TEST_NEQ(handle2.get(), cusolverDnHandle_t{});

        cu::cusolver_handle handle3{handle};
        cu::cusolver_handle handle4{handle2};

        PIKA_TEST_EQ(handle3.get(), cusolverDnHandle_t{});
        PIKA_TEST_NEQ(handle4.get(), cusolverDnHandle_t{});
        PIKA_TEST_NEQ(handle4.get(), handle2.get());
    }

    {
        // Equality is based on the underlying handle.
        cu::cusolver_handle handle1{stream};
        cu::cusolver_handle handle2{stream};
        cu::cusolver_handle handle3{stream};

        PIKA_TEST_NEQ(handle1.get_stream(), whip::stream_t{0});
        PIKA_TEST_NEQ(handle2.get_stream(), whip::stream_t{0});
        PIKA_TEST_NEQ(handle3.get_stream(), whip::stream_t{0});
        PIKA_TEST_EQ(handle1.get_stream(), stream.get());
        PIKA_TEST_EQ(handle2.get_stream(), stream.get());
        PIKA_TEST_EQ(handle3.get_stream(), stream.get());
        PIKA_TEST_NEQ(handle1, handle2);
        PIKA_TEST_NEQ(handle1, handle3);

        cu::cusolver_handle handle4{std::move(handle1)};
        cu::cusolver_handle handle5{std::move(handle2)};
        cu::cusolver_handle handle6{std::move(handle3)};

        PIKA_TEST_EQ(handle1, handle2);
        PIKA_TEST_EQ(handle1, handle3);
        PIKA_TEST_NEQ(handle4, handle5);
        PIKA_TEST_NEQ(handle4, handle6);
    }

    {
        // Equality is based on the underlying handle.
        cu::cusolver_handle handle1{};
        cu::cusolver_handle handle2{};
        cu::cusolver_handle handle3{};

        handle1.set_stream(stream);
        handle2.set_stream(stream);
        handle3.set_stream(stream);

        PIKA_TEST_NEQ(handle1.get_stream(), whip::stream_t{0});
        PIKA_TEST_NEQ(handle2.get_stream(), whip::stream_t{0});
        PIKA_TEST_NEQ(handle3.get_stream(), whip::stream_t{0});
        PIKA_TEST_EQ(handle1.get_stream(), stream.get());
        PIKA_TEST_EQ(handle2.get_stream(), stream.get());
        PIKA_TEST_EQ(handle3.get_stream(), stream.get());
        PIKA_TEST_NEQ(handle1, handle2);
        PIKA_TEST_NEQ(handle1, handle3);

        cu::cusolver_handle handle4{std::move(handle1)};
        cu::cusolver_handle handle5{std::move(handle2)};
        cu::cusolver_handle handle6{std::move(handle3)};

        PIKA_TEST_EQ(handle1, handle2);
        PIKA_TEST_EQ(handle1, handle3);
        PIKA_TEST_NEQ(handle4, handle5);
        PIKA_TEST_NEQ(handle4, handle6);
    }

    // This example is adapted from
    // https://docs.nvidia.com/cuda/cusolver/index.html#lu_examples
    {
        int const m = 3;
        int const lda = m;
        int const ldb = m;

        //       | 1 2 3  |
        //   A = | 4 5 6  |
        //       | 7 8 10 |
        //
        // A = L*U
        //       | 1 0 0 |      | 1  2  3 |
        //   L = | 4 1 0 |, U = | 0 -3 -6 |
        //       | 7 2 1 |      | 0  0  1 |
        double A[lda * m] = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0};
        double B[m] = {1.0, 2.0, 3.0};
        double X[m];        /* X = A\B */
        double LU[lda * m]; /* L and U */
        int info = 0;       /* host copy of error info */

        double* d_A = nullptr; /* device copy of A */
        double* d_B = nullptr; /* device copy of B */
        int* d_info = nullptr; /* error info */
        int* d_piv = nullptr;  /* device vector of pivot indices */
#if defined(PIKA_HAVE_CUDA)
        int lwork = 0;            /* size of workspace */
        double* d_work = nullptr; /* device workspace for getrf */
#endif

        std::cout << "example of getrf \n";
        std::cout << "compute A = L*U (not numerically stable)\n";
        std::cout << "A = \n";
        print_matrix(m, m, A, lda, "A");
        std::cout << "=====\n";
        std::cout << "B = \n";
        print_matrix(m, 1, B, ldb, "B");
        std::cout << "=====\n";

        // step 1: create cusolver handle, bind a stream
        cu::cusolver_handle handle{stream};

        // step 2: copy A to device
        whip::malloc(&d_A, sizeof(double) * lda * m);
        whip::malloc(&d_B, sizeof(double) * m);
        whip::malloc(&d_info, sizeof(int));
        whip::malloc(&d_piv, m * sizeof(int));

        whip::memcpy(d_A, A, sizeof(double) * lda * m, whip::memcpy_host_to_device);
        whip::memcpy(d_B, B, sizeof(double) * m, whip::memcpy_host_to_device);

        // step 3: query working space of getrf
#if defined(PIKA_HAVE_CUDA)
        cu::check_cusolver_error(cusolverDnDgetrf_bufferSize(handle.get(), m, m, d_A, lda, &lwork));
        whip::malloc(&d_work, sizeof(double) * lwork);
#endif

        // step 4: LU factorization
#if defined(PIKA_HAVE_CUDA)
        cu::check_cusolver_error(
            cusolverDnDgetrf(handle.get(), m, m, d_A, lda, d_work, d_piv, d_info));
#else
        cu::check_cusolver_error(cusolverDnDgetrf(handle.get(), m, m, d_A, lda, d_piv, d_info));
#endif
        whip::device_synchronize();
        whip::memcpy(LU, d_A, sizeof(double) * lda * m, whip::memcpy_device_to_host);
        whip::memcpy(&info, d_info, sizeof(int), whip::memcpy_device_to_host);

        PIKA_TEST_EQ(info, 0);

        std::cout << "L and U = \n";
        print_matrix(m, m, LU, lda, "LU");
        std::cout << "=====\n";
        // step 5: solve A*X = B
        //       | 1 |       | -0.3333 |
        //   B = | 2 |,  X = |  0.6667 |
        //       | 3 |       |  0      |
#if defined(PIKA_HAVE_CUDA)
        cu::check_cusolver_error(cusolverDnDgetrs(handle.get(), CUBLAS_OP_N, m, 1, /* nrhs */
            d_A, lda, d_piv, d_B, ldb, d_info));
#else
        cu::check_cusolver_error(cusolverDnDgetrs(handle.get(), CUBLAS_OP_N, m, 1, /* nrhs */
            d_A, lda, d_piv, d_B, ldb));
#endif
        whip::device_synchronize();
        whip::memcpy(X, d_B, sizeof(double) * m, whip::memcpy_device_to_host);

        std::cout << "X = \n";
        print_matrix(m, 1, X, ldb, "X");
        std::cout << "=====\n";

        /* free resources */
        whip::free(d_A);
        whip::free(d_B);
        whip::free(d_info);
        whip::free(d_piv);
#if defined(PIKA_HAVE_CUDA)
        whip::free(d_work);
#endif
    }

    // This would test that a cuSOLVER call gives
    // CUSOLVER_STATUS_NOT_INITIALIZED when used with a default-constructed or
    // moved-from handle. However, cuSOLVER segfaults instead so we skip the
    // test here.
}
