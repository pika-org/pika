//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/assert.hpp>
#include <pika/async_cuda/cuda_stream.hpp>
#include <pika/concurrency/cache_line_data.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/datastructures/optional.hpp>

#include <atomic>
#include <cstddef>
#include <iosfwd>
#include <memory>
#include <vector>

namespace pika::cuda::experimental {
    /// A pool of CUDA streams, used for scheduling work on a CUDA device.
    ///
    /// The pool initializes a set of CUDA (thread-local) streams on
    /// construction and provides access to the streams in a round-robin
    /// fashion. The pool is movable and copyable with reference semantics.
    /// Copies of a pool still refer to the original pool of streams.
    class cuda_pool
    {
    private:
        struct streams_holder
        {
            std::size_t const num_streams_per_thread;
            std::size_t const concurrency;
            std::vector<cuda_stream> streams;
            std::vector<pika::util::cache_aligned_data<std::size_t>>
                active_stream_indices;

            PIKA_EXPORT streams_holder(int device,
                std::size_t num_streams_per_thread,
                pika::threads::thread_priority);
            streams_holder(streams_holder&&) = delete;
            streams_holder(streams_holder const&) = delete;
            streams_holder& operator=(streams_holder&&) = delete;
            streams_holder& operator=(streams_holder const&) = delete;

            PIKA_EXPORT cuda_stream const& get_next_stream();
        };

        struct pool_data
        {
            int device;
            streams_holder normal_priority_streams;
            streams_holder high_priority_streams;

            PIKA_EXPORT pool_data(int device,
                std::size_t num_normal_priority_streams_per_thread,
                std::size_t num_high_priority_streams_per_thread);
            pool_data(pool_data&&) = delete;
            pool_data(pool_data const&) = delete;
            pool_data& operator=(pool_data&&) = delete;
            pool_data& operator=(pool_data const&) = delete;
        };

        std::shared_ptr<pool_data> data;

    public:
        PIKA_EXPORT explicit cuda_pool(int device = 0,
            std::size_t num_normal_priority_streams_per_thread = 3,
            std::size_t num_high_priority_streams_per_thread = 3);
        cuda_pool(cuda_pool&&) = default;
        cuda_pool(cuda_pool const&) = default;
        cuda_pool& operator=(cuda_pool&&) = default;
        cuda_pool& operator=(cuda_pool const&) = default;

        PIKA_EXPORT bool valid() const noexcept;
        PIKA_EXPORT explicit operator bool() noexcept;
        PIKA_EXPORT cuda_stream const& get_next_stream(
            pika::threads::thread_priority priority =
                pika::threads::thread_priority::normal);

        /// \cond NOINTERNAL
        friend bool operator==(
            cuda_pool const& lhs, cuda_pool const& rhs) noexcept
        {
            return lhs.data == rhs.data;
        }

        friend bool operator!=(
            cuda_pool const& lhs, cuda_pool const& rhs) noexcept
        {
            return !(lhs == rhs);
        }

        PIKA_EXPORT friend std::ostream& operator<<(
            std::ostream&, cuda_pool const&);
        /// \endcond
    };
}    // namespace pika::cuda::experimental
