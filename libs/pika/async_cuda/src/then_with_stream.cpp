//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/async_cuda/cublas_handle.hpp>
#include <pika/async_cuda/cuda_stream.hpp>
#include <pika/async_cuda/cusolver_handle.hpp>
#include <pika/async_cuda/custom_blas_api.hpp>
#include <pika/async_cuda/then_with_stream.hpp>

namespace pika::cuda::experimental::then_with_stream_detail {
    pika::cuda::experimental::cublas_handle const&
    get_thread_local_cublas_handle(
        cuda_stream const& stream, cublasPointerMode_t pointer_mode)
    {
        static thread_local pika::cuda::experimental::cublas_handle handle{
            stream};

        handle.set_stream(stream);
        handle.set_pointer_mode(pointer_mode);

        return handle;
    }

#if defined(PIKA_HAVE_CUDA)
    pika::cuda::experimental::cusolver_handle const&
    get_thread_local_cusolver_handle(cuda_stream const& stream)
    {
        static thread_local pika::cuda::experimental::cusolver_handle handle{
            stream};

        handle.set_stream(stream);

        return handle;
    }
#endif
}    // namespace pika::cuda::experimental::then_with_stream_detail
