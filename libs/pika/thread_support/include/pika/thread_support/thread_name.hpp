//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#include <string>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
# include <windows.h>
#endif

namespace pika::detail {
    PIKA_EXPORT std::string& get_thread_name_internal();
    PIKA_EXPORT std::string get_thread_name();
    PIKA_EXPORT void set_thread_name(std::string_view name, std::string_view short_name);
}    // namespace pika::detail
