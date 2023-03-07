//  Copyright (c) 2021 ETH Zurich
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
    PIKA_EXPORT std::uint32_t get_env_var(const char* s, std::uint32_t def);
}    // namespace pika::detail
