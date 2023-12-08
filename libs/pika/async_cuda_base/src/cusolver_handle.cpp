//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/async_cuda_base/cuda_device_scope.hpp>
#include <pika/async_cuda_base/cuda_stream.hpp>
#include <pika/async_cuda_base/cusolver_exception.hpp>
#include <pika/async_cuda_base/cusolver_handle.hpp>
#include <pika/async_cuda_base/custom_blas_api.hpp>

#include <whip.hpp>

namespace pika::cuda::experimental {
    cusolverDnHandle_t cusolver_handle::create_handle(int device, whip::stream_t stream)
    {
        cuda_device_scope d{device};
        cusolverDnHandle_t handle;

        check_cusolver_error(cusolverDnCreate(&handle));
        check_cusolver_error(cusolverDnSetStream(handle, stream));

        return handle;
    }

    cusolver_handle::cusolver_handle()
      : device(0)
      , stream(0)
      , handle(create_handle(this->device, this->stream))
    {
    }

    cusolver_handle::cusolver_handle(cuda_stream const& stream)
      : device(stream.get_device())
      , stream(stream.get())
      , handle(create_handle(device, this->stream))
    {
    }

    cusolver_handle::cusolver_handle(cusolver_handle&& other) noexcept
      : device(other.device)
      , stream(other.stream)
      , handle(other.handle)
    {
        other.stream = 0;
        other.handle = 0;
        other.device = 0;
    }

    cusolver_handle& cusolver_handle::operator=(cusolver_handle&& other) noexcept
    {
        stream = other.stream;
        handle = other.handle;
        device = other.device;

        other.stream = 0;
        other.handle = 0;
        other.device = 0;

        return *this;
    }

    cusolver_handle::cusolver_handle(cusolver_handle const& other)
      : device(other.device)
      , stream(other.stream)
      , handle(other.handle != 0 ? create_handle(device, stream) : 0)
    {
    }

    cusolver_handle& cusolver_handle::operator=(cusolver_handle const& other)
    {
        device = other.device;
        stream = other.stream;
        handle = other.handle != 0 ? create_handle(device, stream) : 0;

        return *this;
    }

    cusolver_handle::~cusolver_handle()
    {
        if (handle != 0) { check_cusolver_error(cusolverDnDestroy(handle)); }
    }

    cusolverDnHandle_t cusolver_handle::get() const noexcept { return handle; }

    int cusolver_handle::get_device() const noexcept { return device; }

    whip::stream_t cusolver_handle::get_stream() const noexcept { return stream; }

    void cusolver_handle::set_stream(cuda_stream const& stream)
    {
        check_cusolver_error(cusolverDnSetStream(handle, stream.get()));
        this->stream = stream.get();
    }
}    // namespace pika::cuda::experimental
