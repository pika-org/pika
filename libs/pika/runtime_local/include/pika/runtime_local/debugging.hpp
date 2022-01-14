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
    /// Attaches a debugger if \c category is equal to the configuration entry
    /// pika.attach-debugger.
    void PIKA_EXPORT may_attach_debugger(std::string const& category);
}}    // namespace pika::util
