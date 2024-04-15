//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file pika/runtime/get_os_thread_count.hpp

#pragma once

#include <pika/config.hpp>

#include <cstddef>

namespace pika {
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the number of OS-threads running in the runtime instance
    ///        the current pika-thread is associated with.
    PIKA_EXPORT std::size_t get_os_thread_count();

}    // namespace pika
