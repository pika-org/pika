////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <pika/config.hpp>

#include <cstdint>

namespace pika {
    enum class runtime_state : std::int8_t
    {
        invalid = -1,
        initialized = 0,
        first_valid_runtime = initialized,
        pre_startup = 1,
        startup = 2,
        pre_main = 3,
        starting = 4,
        running = 5,
        suspended = 6,
        pre_sleep = 7,
        sleeping = 8,
        pre_shutdown = 9,
        shutdown = 10,
        stopping = 11,
        terminating = 12,
        stopped = 13,
        last_valid_runtime = stopped
    };
}
