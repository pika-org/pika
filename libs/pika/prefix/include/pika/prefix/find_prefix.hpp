////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <pika/local/config.hpp>
#include <pika/preprocessor/stringize.hpp>

#include <string>

namespace pika { namespace util {
    // return the full path of the current executable
    PIKA_EXPORT std::string get_executable_filename(
        char const* argv0 = nullptr);
    PIKA_EXPORT std::string get_executable_prefix(
        char const* argv0 = nullptr);
}}    // namespace pika::util
