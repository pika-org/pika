// thread_id.cpp

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details

#include <pika/logging/format/formatters.hpp>

#include <pika/config.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>
#include <fmt/std.h>

#include <memory>
#include <ostream>
#include <type_traits>

namespace pika::util::logging::formatter {

    thread_id::~thread_id() = default;

    struct thread_id_impl : thread_id
    {
        void operator()(std::ostream& to) const override
        {
            auto id = std::this_thread::get_id();
            fmt::print(to, "{}", id);
        }
    };

    std::unique_ptr<thread_id> thread_id::make()
    {
        return std::unique_ptr<thread_id>(new thread_id_impl());
    }

}    // namespace pika::util::logging::formatter
