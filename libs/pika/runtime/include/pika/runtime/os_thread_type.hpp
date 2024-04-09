//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#include <cstdint>
#include <string>
#include <thread>

namespace pika {

    ///////////////////////////////////////////////////////////////////////////
    /// Types of kernel threads registered with the runtime
    enum class os_thread_type
    {
        unknown = -1,
        main_thread = 0,    ///< kernel thread represents main thread
        worker_thread,      ///< kernel thread is used to schedule pika threads
        timer_thread,       ///< kernel is used by timer operations
        custom_thread       ///< kernel is registered by the application
    };

    /// Return a human-readable name representing one of the kernel thread types
    PIKA_EXPORT std::string get_os_thread_type_name(os_thread_type type);

}    // namespace pika
