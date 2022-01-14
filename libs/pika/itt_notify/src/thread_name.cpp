//  Copyright (c) 2019 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/itt_notify/thread_name.hpp>

#include <string>

namespace pika { namespace detail {
    std::string& thread_name()
    {
        static thread_local std::string thread_name_;
        return thread_name_;
    }
}}    // namespace pika::detail
