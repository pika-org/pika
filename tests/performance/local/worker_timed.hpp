//  Copyright (c) 2011-2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//  Copyright (c)      2013 Patricia Grubel
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/modules/timing.hpp>

#include <cstdint>

PIKA_FORCEINLINE void worker_timed(std::uint64_t delay_ns) noexcept
{
    if (delay_ns == 0)
        return;

    auto start = std::chrono::high_resolution_clock::now();

    std::chrono::duration<uint64_t, std::nano> dur;
    while (true)
    {
        // Check if we've reached the specified delay.
        dur = std::chrono::high_resolution_clock::now() - start;
        if (dur.count() >= delay_ns)
            break;
    }
}
