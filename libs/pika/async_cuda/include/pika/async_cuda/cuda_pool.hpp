//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/assert.hpp>
#include <pika/async_cuda_base/cublas_handle.hpp>
#include <pika/async_cuda_base/cuda_stream.hpp>
#include <pika/async_cuda_base/cusolver_handle.hpp>
#include <pika/coroutines/thread_enums.hpp>

#include <fmt/format.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace pika::cuda::experimental {
    /// \brief A locked cuBLAS handle.
    ///
    /// A handle that provides thread-safe access to a \ref cublas_handle. The locked handle is
    /// immovable.
    ///
    /// \note The recommended way to access a handle is through sender adaptors using \ref
    /// cuda_scheduler.
    class locked_cublas_handle
    {
        cublas_handle& handle;
        std::unique_lock<std::mutex> mutex;

    public:
        PIKA_EXPORT locked_cublas_handle(
            cublas_handle& handle, std::unique_lock<std::mutex>&& mutex);
        locked_cublas_handle(locked_cublas_handle&&) = delete;
        locked_cublas_handle(locked_cublas_handle const&) = delete;
        locked_cublas_handle& operator=(locked_cublas_handle&&) = delete;
        locked_cublas_handle& operator=(locked_cublas_handle const&) = delete;

        /// \brief Access the underlying cuBLAS handle.
        ///
        /// \return a reference to the \ref cublas_handle. The returned handle is not thread-safe
        /// and must be used within the lifetime of the \ref locked_cublas_handle.
        PIKA_EXPORT cublas_handle const& get() noexcept;
    };

    /// \brief A locked cuSOLVER handle.
    ///
    /// A handle that provides thread-safe access to a \ref cusolver_handle. The locked handle is
    /// immovable.
    ///
    /// \note The recommended way to access a handle is through sender adaptors using \ref
    /// cuda_scheduler.
    class locked_cusolver_handle
    {
        cusolver_handle& handle;
        std::unique_lock<std::mutex> mutex;

    public:
        PIKA_EXPORT locked_cusolver_handle(
            cusolver_handle& handle, std::unique_lock<std::mutex>&& mutex);
        locked_cusolver_handle(locked_cusolver_handle&&) = delete;
        locked_cusolver_handle(locked_cusolver_handle const&) = delete;
        locked_cusolver_handle& operator=(locked_cusolver_handle&&) = delete;
        locked_cusolver_handle& operator=(locked_cusolver_handle const&) = delete;

        /// \brief Access the underlying cuSOLVER handle.
        ///
        /// \return a reference to the \ref cusolver_handle. The returned handle is not thread-safe
        /// and must be used within the lifetime of the \ref locked_cusolver_handle.
        PIKA_EXPORT cusolver_handle const& get() noexcept;
    };

    /// \brief A pool of CUDA streams, used for scheduling work on a CUDA device.
    ///
    /// The pool initializes a set of CUDA streams on construction and provides access to the
    /// streams in a round-robin fashion. The pool also gives access to cuBLAS and cuSOLVER handles.
    ///
    /// The pool is movable and copyable with reference semantics.  Copies of a pool still refer to
    /// the original pool of streams. A moved-from pool can't be used, except to check if it is
    /// valid with \ref valid().
    ///
    /// The pool is equality comparable and formattable.
    ///
    /// \note The recommended way to access streams and handles from the \ref cuda_pool is through
    /// sender adaptors using \ref cuda_scheduler.
    class cuda_pool
    {
    private:
        struct streams_holder
        {
            std::atomic<std::size_t> stream_index;
            std::vector<cuda_stream> streams;

            PIKA_EXPORT streams_holder(int device, std::size_t num_streams,
                pika::execution::thread_priority, unsigned int flags);
            streams_holder(streams_holder&&) = delete;
            streams_holder(streams_holder const&) = delete;
            streams_holder& operator=(streams_holder&&) = delete;
            streams_holder& operator=(streams_holder const&) = delete;

            PIKA_EXPORT cuda_stream const& get_next_stream();
        };

        struct cublas_handles_holder
        {
            std::atomic<std::size_t> handle_index;
            std::vector<cublas_handle> handles;
            std::vector<std::mutex> handle_mutexes;

            PIKA_EXPORT explicit cublas_handles_holder(std::size_t num_handles);
            cublas_handles_holder(cublas_handles_holder&&) = delete;
            cublas_handles_holder(cublas_handles_holder const&) = delete;
            cublas_handles_holder& operator=(cublas_handles_holder&&) = delete;
            cublas_handles_holder& operator=(cublas_handles_holder const&) = delete;

            PIKA_EXPORT locked_cublas_handle get_locked_handle(
                cuda_stream const& stream, cublasPointerMode_t pointer_mode);
        };

        struct cusolver_handles_holder
        {
            std::atomic<std::size_t> handle_index;
            std::vector<cusolver_handle> handles;
            std::vector<std::mutex> handle_mutexes;

            PIKA_EXPORT explicit cusolver_handles_holder(std::size_t num_handles);
            cusolver_handles_holder(cusolver_handles_holder&&) = delete;
            cusolver_handles_holder(cusolver_handles_holder const&) = delete;
            cusolver_handles_holder& operator=(cusolver_handles_holder&&) = delete;
            cusolver_handles_holder& operator=(cusolver_handles_holder const&) = delete;

            PIKA_EXPORT locked_cusolver_handle get_locked_handle(cuda_stream const& stream);
        };

        struct pool_data
        {
            int device;
            streams_holder normal_priority_streams;
            streams_holder high_priority_streams;
            cublas_handles_holder cublas_handles;
            cusolver_handles_holder cusolver_handles;

            PIKA_EXPORT pool_data(int device, std::size_t num_normal_priority_streams,
                std::size_t num_high_priority_streams, unsigned int flags,
                std::size_t num_cublas_handles, std::size_t num_cusolver_handles);
            pool_data(pool_data&&) = delete;
            pool_data(pool_data const&) = delete;
            pool_data& operator=(pool_data&&) = delete;
            pool_data& operator=(pool_data const&) = delete;
        };

        std::shared_ptr<pool_data> data;

    public:
        /// \brief Construct a pool of CUDA streams and handles.
        ///
        /// \param device the CUDA device used for scheduling work
        /// \param num_normal_priority_streams_per_thread the number of normal priority streams per
        /// thread
        /// \param num_high_priority_streams_per_thread the number of high priority streams per
        /// thread
        /// \param flags flags used to construct CUDA streams
        /// \param num_cublas_handles the number of cuBLAS handles to create for the whole pool
        /// \param num_cusolver_handles the number of cuSOLVER handles to create for the whole pool
        PIKA_EXPORT explicit cuda_pool(int device = 0, std::size_t num_normal_priority_streams = 32,
            std::size_t num_high_priority_streams = 32, unsigned int flags = 0,
            std::size_t num_cublas_handles = 16, std::size_t num_cusolver_handles = 16);

        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        cuda_pool(cuda_pool&&) = default;
        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        cuda_pool(cuda_pool const&) = default;
        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        cuda_pool& operator=(cuda_pool&&) = default;
        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        cuda_pool& operator=(cuda_pool const&) = default;

        /// \brief Check if the pool is valid.
        ///
        /// \return true if the pool refers to a valid pool, false otherwise (e.g. if the pool has
        /// been moved out from).
        PIKA_EXPORT bool valid() const noexcept;

        /// \brief Check if the pool is valid.
        ///
        /// See \ref valid().
        PIKA_EXPORT explicit operator bool() noexcept;

        /// \brief Get a reference to the next CUDA stream.
        ///
        /// \note The recommended way to access a stream is through a \ref cuda_scheduler.
        ///
        /// \param priority the priority of the stream.
        PIKA_EXPORT cuda_stream const& get_next_stream(
            pika::execution::thread_priority priority = pika::execution::thread_priority::normal);

        /// \brief Get a locked cuBLAS handle.
        ///
        /// \note The recommended way to access a handle is through a \ref cuda_scheduler.
        ///
        /// \param stream the CUDA stream to use with the cuBLAS handle.
        ///
        /// \return a locked cuBLAS handle, which is released for reuse on destruction.
        PIKA_EXPORT locked_cublas_handle get_cublas_handle(
            cuda_stream const& stream, cublasPointerMode_t pointer_mode);

        /// \brief Get a locked cuSOLVER handle.
        ///
        /// \note The recommended way to access a handle is through a \ref cuda_scheduler.
        ///
        /// \param stream the CUDA stream to use with the cuSOLVER handle.
        ///
        /// \return a locked cuSOLVER handle, which is released for reuse on destruction.
        PIKA_EXPORT locked_cusolver_handle get_cusolver_handle(cuda_stream const& stream);

        /// \cond NOINTERNAL
        friend bool operator==(cuda_pool const& lhs, cuda_pool const& rhs) noexcept
        {
            return lhs.data == rhs.data;
        }

        friend bool operator!=(cuda_pool const& lhs, cuda_pool const& rhs) noexcept
        {
            return !(lhs == rhs);
        }

        friend struct fmt::formatter<pika::cuda::experimental::cuda_pool>;
        /// \endcond
    };
}    // namespace pika::cuda::experimental

template <>
struct fmt::formatter<pika::cuda::experimental::cuda_pool> : fmt::formatter<std::string>
{
    template <typename FormatContext>
    auto format(pika::cuda::experimental::cuda_pool const& pool, FormatContext& ctx) const
    {
        bool valid{pool.data};
        auto num_high_priority_streams =
            valid ? pool.data->high_priority_streams.streams.size() : 0;
        auto num_normal_priority_streams =
            valid ? pool.data->normal_priority_streams.streams.size() : 0;
        auto num_cublas_handles = valid ? pool.data->cublas_handles.handles.size() : 0;
        auto num_cusolver_handles = valid ? pool.data->cusolver_handles.handles.size() : 0;
        return fmt::formatter<std::string>::format(
            fmt::format("cuda_pool({}, num_high_priority_streams = {}, "
                        "num_normal_priority_streams = {}, num_cublas_handles = {}, "
                        "num_cusolver_handles = {})",
                fmt::ptr(pool.data.get()), num_high_priority_streams, num_normal_priority_streams,
                num_cublas_handles, num_cusolver_handles),
            ctx);
    }
};
