//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/coroutines/thread_enums.hpp>

#include <fmt/format.h>
#include <whip.hpp>

#include <string>

namespace pika::cuda::experimental {
    /// \brief RAII wrapper for a CUDA stream.
    ///
    /// An RAII wrapper for a CUDA stream which creates a stream on construction and destroys it on
    /// destruction. It is movable and copyable. A moved-from stream holds the default stream. A
    /// copied stream uses the properties from the given stream and creates a new stream.
    ///
    /// Equality comparable and formattable.
    ///
    /// When accessing the underlying stream, [whip](https://github.com/eth-cscs/whip) is used for
    /// compatibility with CUDA and HIP.
    ///
    /// \note The recommended way to access a stream is through sender adaptors using \ref
    /// cuda_scheduler.
    class cuda_stream
    {
    private:
        int device;
        pika::execution::thread_priority priority;
        unsigned int flags;
        whip::stream_t stream{};

        struct priorities
        {
            int least;
            int greatest;
        };

        static PIKA_EXPORT priorities get_available_priorities();
        static PIKA_EXPORT whip::stream_t create_stream(
            int device, pika::execution::thread_priority priority, unsigned int flags);

    public:
        /// \brief Construct a new stream with the given device and priority.
        ///
        /// \param device The device to create the stream on.
        /// \param priority The priority of the stream. The mapping from \ref thread_priority to
        /// CUDA stream priorities is undefined, except that the order is preserved, allowing for
        /// different \ref thread_priority to map to the same CUDA priority.
        /// \param flags Flags to pass to the CUDA stream creation.
        PIKA_EXPORT explicit cuda_stream(int device = 0,
            pika::execution::thread_priority priority = pika::execution::thread_priority::default_,
            unsigned int flags = 0);
        PIKA_EXPORT ~cuda_stream();
        PIKA_EXPORT cuda_stream(cuda_stream&&) noexcept;
        PIKA_EXPORT cuda_stream& operator=(cuda_stream&&) noexcept;
        PIKA_EXPORT cuda_stream(cuda_stream const&);
        PIKA_EXPORT cuda_stream& operator=(cuda_stream const&);

        /// \brief Get the underlying stream.
        ///
        /// The stream is still owned by the \ref cuda_stream and must not be manually released.
        PIKA_EXPORT whip::stream_t get() const noexcept;

        /// \brief Get the device of the stream.
        PIKA_EXPORT int get_device() const noexcept;

        /// \brief Get the priority of the stream.
        PIKA_EXPORT pika::execution::thread_priority get_priority() const noexcept;

        /// brief Get the flags of the stream.
        PIKA_EXPORT unsigned int get_flags() const noexcept;

        /// \cond NOINTERNAL
        friend bool operator==(cuda_stream const& lhs, cuda_stream const& rhs) noexcept
        {
            return lhs.stream == rhs.stream;
        }

        friend bool operator!=(cuda_stream const& lhs, cuda_stream const& rhs) noexcept
        {
            return !(lhs == rhs);
        }
        /// \endcond
    };
}    // namespace pika::cuda::experimental

template <>
struct fmt::formatter<pika::cuda::experimental::cuda_stream> : fmt::formatter<std::string>
{
    template <typename FormatContext>
    auto format(pika::cuda::experimental::cuda_stream const& stream, FormatContext& ctx) const
    {
        return fmt::formatter<std::string>::format(
            fmt::format("cuda_stream({})", fmt::ptr(stream.get())), ctx);
    }
};
