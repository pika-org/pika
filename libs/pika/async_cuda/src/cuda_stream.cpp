//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/async_cuda/cuda_device_scope.hpp>
#include <pika/async_cuda/cuda_exception.hpp>
#include <pika/async_cuda/cuda_stream.hpp>
#include <pika/async_cuda/custom_gpu_api.hpp>
#include <pika/coroutines/thread_enums.hpp>

#include <ostream>

namespace pika::cuda::experimental {
    cuda_stream::priorities cuda_stream::get_available_priorities()
    {
        cuda_stream::priorities p;

        check_cuda_error(
            cudaDeviceGetStreamPriorityRange(&p.least, &p.greatest));

        return p;
    }

    cudaStream_t cuda_stream::create_stream(
        int device, pika::threads::thread_priority priority)
    {
        cuda_device_scope d{device};
        auto p = get_available_priorities();

        cudaStream_t stream;
        if (priority <= pika::threads::thread_priority::normal)
        {
            check_cuda_error(cudaStreamCreateWithPriority(
                &stream, cudaStreamNonBlocking, p.least));
        }
        else
        {
            check_cuda_error(cudaStreamCreateWithPriority(
                &stream, cudaStreamNonBlocking, p.greatest));
        }

        return stream;
    }

    cuda_stream::cuda_stream(
        int device, pika::threads::thread_priority priority)
      : device(device)
      , priority(priority)
      , stream(create_stream(device, priority))
    {
    }

    cuda_stream::cuda_stream(cuda_stream&& other) noexcept
      : device(other.device)
      , priority(other.priority)
      , stream(other.stream)
    {
        other.device = 0;
        other.priority = pika::threads::thread_priority::default_;
        other.stream = 0;
    }

    cuda_stream& cuda_stream::operator=(cuda_stream&& other) noexcept
    {
        device = other.device;
        priority = other.priority;
        stream = other.stream;

        other.device = 0;
        other.priority = pika::threads::thread_priority::default_;
        other.stream = 0;

        return *this;
    }

    cuda_stream::cuda_stream(cuda_stream const& other)
      : device(other.device)
      , priority(other.priority)
      , stream(other.stream != 0 ? create_stream(device, priority) : 0)
    {
    }

    cuda_stream& cuda_stream::operator=(cuda_stream const& other)
    {
        device = other.device;
        priority = other.priority;
        stream = other.stream != 0 ? create_stream(device, priority) : 0;

        return *this;
    }

    cuda_stream::~cuda_stream()
    {
        if (stream != 0)
        {
            check_cuda_error(cudaStreamDestroy(stream));
        }
    }

    int cuda_stream::get_device() const noexcept
    {
        return device;
    }

    pika::threads::thread_priority cuda_stream::get_priority() const noexcept
    {
        return priority;
    }

    cudaStream_t cuda_stream::get() const noexcept
    {
        return stream;
    }

    std::ostream& operator<<(std::ostream& os, cuda_stream const& stream)
    {
        os << "cuda_stream(" << stream.get() << ")";
        return os;
    }
}    // namespace pika::cuda::experimental
