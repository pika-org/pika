//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/coroutines/thread_enums.hpp>

#include <whip.hpp>

#include <iosfwd>

namespace pika::cuda::experimental {
    /// RAII wrapper for a CUDA stream.
    ///
    /// An RAII wrapper for a CUDA stream which creates a stream on construction
    /// and destroys it on destruction. Is movable and copiable. A moved-from
    /// stream holds the default stream. A copied stream uses the properties
    /// from the given stream and creates a new stream.
    class cuda_stream
    {
    private:
        int device;
        pika::execution::thread_priority priority;
        unsigned int flags;
        whip::stream_t stream = 0;

        struct priorities
        {
            int least;
            int greatest;
        };

        static PIKA_EXPORT priorities get_available_priorities();
        static PIKA_EXPORT whip::stream_t create_stream(int device,
            pika::execution::thread_priority priority, unsigned int flags);

    public:
        PIKA_EXPORT explicit cuda_stream(int device = 0,
            pika::execution::thread_priority priority =
                pika::execution::thread_priority::default_,
            unsigned int flags = 0);
        PIKA_EXPORT ~cuda_stream();
        PIKA_EXPORT cuda_stream(cuda_stream&&) noexcept;
        PIKA_EXPORT cuda_stream& operator=(cuda_stream&&) noexcept;
        PIKA_EXPORT cuda_stream(cuda_stream const&);
        PIKA_EXPORT cuda_stream& operator=(cuda_stream const&);

        PIKA_EXPORT whip::stream_t get() const noexcept;
        PIKA_EXPORT int get_device() const noexcept;
        PIKA_EXPORT pika::execution::thread_priority
        get_priority() const noexcept;
        PIKA_EXPORT unsigned int get_flags() const noexcept;

        /// \cond NOINTERNAL
        friend bool operator==(
            cuda_stream const& lhs, cuda_stream const& rhs) noexcept
        {
            return lhs.stream == rhs.stream;
        }

        friend bool operator!=(
            cuda_stream const& lhs, cuda_stream const& rhs) noexcept
        {
            return !(lhs == rhs);
        }

        PIKA_EXPORT friend std::ostream& operator<<(
            std::ostream&, cuda_stream const&);
        /// \endcond
    };
}    // namespace pika::cuda::experimental
