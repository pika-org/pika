//  Copyright (c) 2018 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/future.hpp>

#include <pika/async_cuda/cuda_executor.hpp>
#include <pika/async_cuda/custom_gpu_api.hpp>

#include <cstddef>

__global__ void saxpy(int n, float a, float* x, float* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

void launch_saxpy_kernel(pika::cuda::experimental::cuda_executor& cudaexec,
    unsigned int& blocks, unsigned int& threads, void** args)
{
    // Invoking pika::apply with cudaLaunchKernel<void> directly result in an
    // error for NVCC with gcc configuration
#ifdef PIKA_HAVE_HIP
    auto launch_kernel = cudaLaunchKernel;
#else
    auto launch_kernel = cudaLaunchKernel<void>;
#endif
    pika::apply(cudaexec, launch_kernel, reinterpret_cast<const void*>(&saxpy),
        dim3(blocks), dim3(threads), args, std::size_t(0));
}
