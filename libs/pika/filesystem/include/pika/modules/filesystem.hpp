//  Copyright (c) 2019 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  pikainspect:nodeprecatedinclude:boost/filesystem.hpp

/// \file
/// This file provides a few additional filesystem utilities on top of those
/// provided by <filesystem>.

#pragma once

#include <pika/config.hpp>

#include <filesystem>
#include <string>
#include <system_error>

namespace pika::detail::filesystem {
    inline std::filesystem::path initial_path()
    {
        static std::filesystem::path ip = std::filesystem::current_path();
        return ip;
    }

    inline std::string basename(std::filesystem::path const& p)
    {
        return p.stem().string();
    }

    inline std::filesystem::path canonical(
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

    inline std::filesystem::path canonical(std::filesystem::path const& p,
        std::filesystem::path const& base, std::error_code& ec)
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
