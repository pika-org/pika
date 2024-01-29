//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/async_cuda/cublas_exception.hpp>
#include <pika/async_cuda/cublas_handle.hpp>
#include <pika/async_cuda/cuda_device_scope.hpp>
#include <pika/async_cuda/cuda_stream.hpp>
#include <pika/async_cuda/custom_blas_api.hpp>

#include <whip.hpp>

namespace pika::cuda::experimental {
    cublasHandle_t cublas_handle::create_handle(int device, whip::stream_t stream)
    {
        cuda_device_scope d{device};
        cublasHandle_t handle;

        check_cublas_error(cublasCreate(&handle));
        check_cublas_error(cublasSetStream(handle, stream));

        return handle;
    }

    cublas_handle::cublas_handle()
      : device(0)
      , stream(0)
      , handle(create_handle(this->device, this->stream))
    {
    }

    cublas_handle::cublas_handle(cuda_stream const& stream)
      : device(stream.get_device())
      , stream(stream.get())
      , handle(create_handle(device, this->stream))
    {
    }

    cublas_handle::cublas_handle(cublas_handle&& other) noexcept
      : device(other.device)
      , stream(other.stream)
      , handle(other.handle)
    {
        other.device = 0;
        other.stream = 0;
        other.handle = 0;
    }

    cublas_handle& cublas_handle::operator=(cublas_handle&& other) noexcept
    {
        device = other.device;
        stream = other.stream;
        handle = other.handle;

        other.device = 0;
        other.stream = 0;
        other.handle = 0;

        return *this;
    }

    cublas_handle::cublas_handle(cublas_handle const& other)
      : device(other.device)
      , stream(other.stream)
      , handle(other.handle != 0 ? create_handle(device, stream) : 0)
    {
    }

    cublas_handle& cublas_handle::operator=(cublas_handle const& other)
    {
        device = other.device;
        stream = other.stream;
        handle = other.handle != 0 ? create_handle(device, stream) : 0;

        return *this;
    }

    cublas_handle::~cublas_handle()
    {
        if (handle != 0) { check_cublas_error(cublasDestroy(handle)); }
    }

    cublasHandle_t cublas_handle::get() const noexcept { return handle; }

    int cublas_handle::get_device() const noexcept { return device; }

    whip::stream_t cublas_handle::get_stream() const noexcept { return stream; }

    void cublas_handle::set_stream(cuda_stream const& stream)
    {
        check_cublas_error(cublasSetStream(handle, stream.get()));
        this->stream = stream.get();
    }

    void cublas_handle::set_pointer_mode(cublasPointerMode_t pointer_mode)
    {
        check_cublas_error(cublasSetPointerMode(handle, pointer_mode));
    }
}    // namespace pika::cuda::experimental
