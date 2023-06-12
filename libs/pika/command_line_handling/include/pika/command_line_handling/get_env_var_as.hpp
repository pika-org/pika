//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config/export_definitions.hpp>

#include <cstdlib>
#include <string>

namespace pika::detail {

    /// from env var name 's' get value if well-formed, otherwise return default
    template <typename T>
    PIKA_EXPORT T get_env_var_as(const char* s, T def);
}    // namespace pika::detail
