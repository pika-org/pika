//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/runtime/os_thread_type.hpp>

#include <string>

namespace pika {

    namespace detail {
        // clang-format off
        static char const* const thread_type_names[] = {
            "unknown",
            "main-thread",
            "worker-thread",
            "io-thread",
            "timer-thread",
            "custom-thread"
        };
        // clang-format on
    }    // namespace detail

    std::string get_os_thread_type_name(os_thread_type type)
    {
        int idx = static_cast<int>(type);
        if (idx < -1 || idx > static_cast<int>(os_thread_type::custom_thread)) { idx = -1; }
        return detail::thread_type_names[idx + 1];
    }
}    // namespace pika
