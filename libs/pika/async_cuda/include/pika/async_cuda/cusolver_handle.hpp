//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/async_cuda/cuda_exception.hpp>
#include <pika/async_cuda/cuda_stream.hpp>
#include <pika/async_cuda/custom_gpu_api.hpp>
#include <pika/async_cuda/custom_lapack_api.hpp>

#include <iosfwd>

namespace pika::cuda::experimental {
    /// RAII wrapper for a cuSOLVER handle.
    class cusolver_handle
    {
    private:
        int device;
        cudaStream_t stream{};
        cusolverDnHandle_t handle{};

        static PIKA_EXPORT cusolverDnHandle_t create_handle(
            int device, cudaStream_t stream);

    public:
        PIKA_EXPORT cusolver_handle();
        PIKA_EXPORT explicit cusolver_handle(cuda_stream const& stream);
        PIKA_EXPORT ~cusolver_handle();
        PIKA_EXPORT cusolver_handle(cusolver_handle&&) noexcept;
        PIKA_EXPORT cusolver_handle& operator=(cusolver_handle&&) noexcept;
        PIKA_EXPORT cusolver_handle(cusolver_handle const&);
        PIKA_EXPORT cusolver_handle& operator=(cusolver_handle const&);

        PIKA_EXPORT cusolverDnHandle_t get() const noexcept;
        PIKA_EXPORT int get_device() const noexcept;
        PIKA_EXPORT cudaStream_t get_stream() const noexcept;

        PIKA_EXPORT void set_stream(cuda_stream const& stream);

        /// \cond NOINTERNAL
        friend bool operator==(
            cusolver_handle const& lhs, cusolver_handle const& rhs) noexcept
        {
            return lhs.handle == rhs.handle;
        }

        friend bool operator!=(
            cusolver_handle const& lhs, cusolver_handle const& rhs) noexcept
        {
            return !(lhs == rhs);
        }

        PIKA_EXPORT friend std::ostream& operator<<(
            std::ostream&, cusolver_handle const&);
        /// \endcond
    };
}    // namespace pika::cuda::experimental
