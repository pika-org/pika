//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/command_line_handling/get_env_var_as.hpp>
#include <pika/config/export_definitions.hpp>
#include <pika/modules/logging.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace pika::detail {

    template <typename T>
    PIKA_EXPORT T get_env_var_as(const char* s, T def)
    {
        T val = def;
        char* env = std::getenv(s);
        if (env)
        {
            try
            {
                std::istringstream temp(env);
                temp >> val;
            }
            catch (...)
            {
                val = def;
                std::cerr << "get_env_var_as - invalid"
                          << " " << s << " " << def << std::endl;
                LERR_(error) << "get_env_var_as - invalid" << s << val;
            }
            LBT_(debug) << "get_env_var_as " << s << val;
        }
        return val;
    }

    template PIKA_EXPORT std::uint32_t get_env_var_as(const char* s, std::uint32_t def);
    template PIKA_EXPORT std::uint64_t get_env_var_as(const char* s, std::uint64_t def);
    template PIKA_EXPORT std::int32_t get_env_var_as(const char* s, std::int32_t def);
    template PIKA_EXPORT std::int64_t get_env_var_as(const char* s, std::int64_t def);
    template PIKA_EXPORT float get_env_var_as(const char* s, float def);
    template PIKA_EXPORT double get_env_var_as(const char* s, double def);
    template PIKA_EXPORT std::string get_env_var_as(const char* s, std::string def);

}    // namespace pika::detail
