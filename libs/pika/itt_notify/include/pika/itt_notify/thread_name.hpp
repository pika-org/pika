//  Copyright (c) 2019 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <string>
#endif

namespace pika::detail {
    /// Helper utility to set and store a name for the current operating system
    /// thread. Returns a reference to the name for the current thread.
    PIKA_EXPORT std::string& thread_name();
}    // namespace pika::detail
