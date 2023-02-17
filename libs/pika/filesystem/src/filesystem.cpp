//  Copyright (c) 2019-2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/detail/filesystem.hpp>

#include <filesystem>
#include <string>
#include <system_error>

namespace pika::detail::filesystem {
    std::filesystem::path initial_path()
    {
        static std::filesystem::path ip = std::filesystem::current_path();
        return ip;
    }

    std::string basename(std::filesystem::path const& p)
    {
        return p.stem().string();
    }

    std::filesystem::path canonical(
        std::filesystem::path const& p, std::filesystem::path const& base)
    {
        if (p.is_relative())
        {
            return canonical(base / p);
        }
        else
        {
            return canonical(p);
        }
    }

    std::filesystem::path canonical(
        std::filesystem::path const& p, std::filesystem::path const& base, std::error_code& ec)
    {
        if (p.is_relative())
        {
            return canonical(base / p, ec);
        }
        else
        {
            return canonical(p, ec);
        }
    }
}    // namespace pika::detail::filesystem
