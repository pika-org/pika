//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/async_cuda_base/cuda_stream.hpp>
#include <pika/async_cuda_base/custom_lapack_api.hpp>

#include <fmt/format.h>
#include <whip.hpp>

#include <string>

namespace pika::cuda::experimental {
    /// \brief RAII wrapper for a cuSOLVER handle.
    ///
    /// An RAII wrapper for a cuBLAS handle which creates a handle on construction and destroys it
    /// on destruction.
    ///
    /// The wrapper is movable and copyable. A moved-from handle can not be used other than to check
    /// for validity with \ref valid(). A copied stream uses the properties from the given handle
    /// and creates a new handle.
    ///
    /// Equality comparable and formattable.
    ///
    /// \note The recommended way to access a handle is through sender adaptors using \ref
    /// cuda_scheduler.
    class cusolver_handle
    {
    private:
        int device;
        whip::stream_t stream{};
        cusolverDnHandle_t handle{};

        static PIKA_EXPORT cusolverDnHandle_t create_handle(int device, whip::stream_t stream);

    public:
        /// \brief Constructs a new cuSOLVER handle with the default stream.
        PIKA_EXPORT cusolver_handle();

        /// \brief Constructs a new cuSOLVER handle with the given stream.
        PIKA_EXPORT explicit cusolver_handle(cuda_stream const& stream);
        PIKA_EXPORT ~cusolver_handle();
        PIKA_EXPORT cusolver_handle(cusolver_handle&&) noexcept;
        PIKA_EXPORT cusolver_handle& operator=(cusolver_handle&&) noexcept;
        PIKA_EXPORT cusolver_handle(cusolver_handle const&);
        PIKA_EXPORT cusolver_handle& operator=(cusolver_handle const&);

        /// \brief Check if the handle is valid.
        ///
        /// \return true if the handle refers to a valid handle, false otherwise (e.g. if the handle
        /// has been moved out from, or it has been default-constructed)
        PIKA_EXPORT bool valid() const noexcept;

        /// \brief Check if the handle is valid.
        ///
        /// See \ref valid().
        PIKA_EXPORT explicit operator bool() const noexcept;

        /// \brief Get the underlying cuSOLVER handle.
        PIKA_EXPORT cusolverDnHandle_t get() const noexcept;

        /// \brief Get the device associated with the cuSOLVER handle.
        PIKA_EXPORT int get_device() const noexcept;

        /// \brief Get the stream associated with the cuSOLVER handle.
        PIKA_EXPORT whip::stream_t get_stream() const noexcept;

        /// \brief Set the stream associated with the cuSOLVER handle.
        PIKA_EXPORT void set_stream(cuda_stream const& stream);

        /// \cond NOINTERNAL
        friend bool operator==(cusolver_handle const& lhs, cusolver_handle const& rhs) noexcept
        {
            return lhs.handle == rhs.handle;
        }

        friend bool operator!=(cusolver_handle const& lhs, cusolver_handle const& rhs) noexcept
        {
            return !(lhs == rhs);
        }
        /// \endcond
    };
}    // namespace pika::cuda::experimental

template <>
struct fmt::formatter<pika::cuda::experimental::cusolver_handle> : fmt::formatter<std::string>
{
    template <typename FormatContext>
    auto format(pika::cuda::experimental::cusolver_handle const& handle, FormatContext& ctx) const
    {
        return fmt::formatter<std::string>::format(
            fmt::format("cusolver_handle({})", fmt::ptr(handle.get())), ctx);
    }
};
