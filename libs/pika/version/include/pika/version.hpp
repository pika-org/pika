//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2013 Adrian Serio
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/config/config_strings.hpp>
#include <pika/config/version.hpp>
#include <pika/modules/format.hpp>
#include <pika/prefix/find_prefix.hpp>
#include <pika/preprocessor/stringize.hpp>

#include <cstdint>
#include <string>
#include <string_view>

///////////////////////////////////////////////////////////////////////////////
namespace pika {
    // Returns the major pika version.
    constexpr std::uint8_t major_version()
    {
        return PIKA_VERSION_MAJOR;
    }

    // Returns the minor pika version.
    constexpr std::uint8_t minor_version()
    {
        return PIKA_VERSION_MINOR;
    }

    // Returns the sub-minor/patch-level pika version.
    constexpr std::uint8_t patch_version()
    {
        return PIKA_VERSION_PATCH;
    }

    // Returns the full pika version.
    constexpr std::uint32_t full_version()
    {
        return PIKA_VERSION_FULL;
    }

    // Returns the full pika version.
    PIKA_EXPORT std::string full_version_as_string();

    // Returns the tag.
    constexpr std::string_view tag()
    {
        return PIKA_VERSION_TAG;
    }

    // Return the pika configuration information.
    PIKA_EXPORT std::string configuration_string();

    // Returns the pika version string.
    PIKA_EXPORT std::string build_string();

    // Returns the pika build type ('Debug', 'Release', etc.)
    constexpr std::string_view build_type()
    {
        return PIKA_PP_STRINGIZE(PIKA_BUILD_TYPE);
    }

    // Returns the pika build date and time
    PIKA_EXPORT std::string build_date_time();

    // Returns the pika full build information string.
    PIKA_EXPORT std::string full_build_string();

    // Returns the copyright string.
    constexpr std::string_view copyright()
    {
        char const* const copyright =
            "pika\n\n"
            "Copyright (c) 2021-2022, ETH Zurich,\n"
            "https://github.com/pika-org/pika\n\n"
            "Distributed under the Boost Software License, "
            "Version 1.0. (See accompanying\n"
            "file LICENSE_1_0.txt or copy at "
            "http://www.boost.org/LICENSE_1_0.txt)\n";
        return copyright;
    }

    // Returns the full version string.
    PIKA_EXPORT std::string complete_version();
}    // namespace pika
