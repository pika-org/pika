//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

#if defined(__bgq__)
#include <hwi/include/bqc/A2_inlines.h>
#endif

#include <chrono>
#include <cstdint>

namespace pika { namespace chrono {

    struct high_resolution_clock
    {
        // This function returns a tick count with a resolution (not
        // precision!) of 1 ns.
        static std::uint64_t now() noexcept
        {
#if defined(__bgq__)
            return GetTimeBase();
#else
            std::chrono::nanoseconds ns =
                std::chrono::steady_clock::now().time_since_epoch();
            return static_cast<std::uint64_t>(ns.count());
#endif
        }

        // This function returns the smallest representable time unit as
        // returned by this clock.
        static constexpr std::uint64_t(min)() noexcept
        {
            typedef std::chrono::duration_values<std::chrono::nanoseconds>
                duration_values;
            return (duration_values::min)().count();
        }

        // This function returns the largest representable time unit as
        // returned by this clock.
        static constexpr std::uint64_t(max)() noexcept
        {
            typedef std::chrono::duration_values<std::chrono::nanoseconds>
                duration_values;
            return (duration_values::max)().count();
        }
    };
}}    // namespace pika::chrono

namespace pika { namespace util {
    using high_resolution_clock PIKA_DEPRECATED_V(0, 1,
        "pika::util::high_resolution_clock is deprecated. Use "
        "pika::chrono::high_resolution_clock instead.") =
        pika::chrono::high_resolution_clock;
}}    // namespace pika::util
