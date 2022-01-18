//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2017      Denis Blank
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <string>

namespace pika { namespace util {
    /// Tries to break an attached debugger, if not supported a loop is
    /// invoked which gives enough time to attach a debugger manually.
    PIKA_EXPORT void attach_debugger();
}}    // namespace pika::util
