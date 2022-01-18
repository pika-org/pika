//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2017      Denis Blank
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/debugging/attach_debugger.hpp>
#include <pika/runtime_local/config_entry.hpp>
#include <pika/runtime_local/debugging.hpp>

#include <string>

namespace pika { namespace util {
    void may_attach_debugger(std::string const& category)
    {
        if (get_config_entry("pika.attach_debugger", "") == category)
        {
            attach_debugger();
        }
    }
}}    // namespace pika::util
