//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/command_line_handling/get_env_var.hpp>
#include <pika/config/export_definitions.hpp>
#include <pika/modules/logging.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace pika::detail {

    PIKA_EXPORT std::uint32_t get_env_var(const char* s, std::uint32_t def)
    {
        std::uint32_t val = def;
        char* env = std::getenv(s);
        if (env)
        {
            try
            {
                val = std::stoi(env);
            }
            catch (...)
            {
                val = def;
                std::cerr << "get_env_value - invalid"
                          << " " << s << " " << def << std::endl;
                LERR_(error) << "get_env_value - invalid" << s << val;
            }
            LBT_(debug) << "get_env_value " << s << val;
        }
        return val;
    }

}    // namespace pika::detail
