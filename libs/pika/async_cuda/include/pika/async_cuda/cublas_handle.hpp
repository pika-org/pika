//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/async_cuda/cuda_stream.hpp>
#include <pika/async_cuda/custom_blas_api.hpp>

#include <whip.hpp>

#include <iosfwd>

namespace pika::cuda::experimental {
    /// RAII wrapper for a cuBLAS handle.
    class cublas_handle
    {
    private:
        int device;
        whip::stream_t stream{};
        cublasHandle_t handle{};

        static PIKA_EXPORT cublasHandle_t create_handle(
            int device, whip::stream_t stream);

    public:
        PIKA_EXPORT cublas_handle();
        PIKA_EXPORT explicit cublas_handle(cuda_stream const& stream);
        PIKA_EXPORT ~cublas_handle();
        PIKA_EXPORT cublas_handle(cublas_handle&&) noexcept;
        PIKA_EXPORT cublas_handle& operator=(cublas_handle&&) noexcept;
        PIKA_EXPORT cublas_handle(cublas_handle const&);
        PIKA_EXPORT cublas_handle& operator=(cublas_handle const&);

        PIKA_EXPORT cublasHandle_t get() const noexcept;
        PIKA_EXPORT int get_device() const noexcept;
        PIKA_EXPORT whip::stream_t get_stream() const noexcept;

        PIKA_EXPORT void set_stream(cuda_stream const& stream);
        PIKA_EXPORT void set_pointer_mode(cublasPointerMode_t pointer_mode);

        /// \cond NOINTERNAL
        friend bool operator==(
            cublas_handle const& lhs, cublas_handle const& rhs) noexcept
        {
            return lhs.handle == rhs.handle;
        }

        friend bool operator!=(
            cublas_handle const& lhs, cublas_handle const& rhs) noexcept
        {
            return !(lhs == rhs);
        }

        PIKA_EXPORT friend std::ostream& operator<<(
            std::ostream&, cublas_handle const&);
        /// \endcond
    };
}    // namespace pika::cuda::experimental
