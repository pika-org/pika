//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2013 Adrian Serio
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

#include <cstdint>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace pika::local {
    // Returns the major pika version.
    PIKA_EXPORT std::uint8_t major_version();

    // Returns the minor pika version.
    PIKA_EXPORT std::uint8_t minor_version();

    // Returns the sub-minor/patch-level pika version.
    PIKA_EXPORT std::uint8_t subminor_version();

    // Returns the full pika version.
    PIKA_EXPORT std::uint32_t full_version();

    // Returns the full pika version.
    PIKA_EXPORT std::string full_version_as_string();

    // Returns the tag.
    PIKA_EXPORT std::string tag();

    // Returns the pika full build information string.
    PIKA_EXPORT std::string full_build_string();

    // Returns the pika version string.
    PIKA_EXPORT std::string build_string();

    // Returns the Boost version string.
    PIKA_EXPORT std::string boost_version();

    // Returns the Boost platform string.
    PIKA_EXPORT std::string boost_platform();

    // Returns the Boost compiler string.
    PIKA_EXPORT std::string boost_compiler();

    // Returns the Boost standard library string.
    PIKA_EXPORT std::string boost_stdlib();

    // Returns the copyright string.
    PIKA_EXPORT std::string copyright();

    // Returns the full version string.
    PIKA_EXPORT std::string complete_version();

    // Returns the pika build type ('Debug', 'Release', etc.)
    PIKA_EXPORT std::string build_type();

    // Returns the pika build date and time
    PIKA_EXPORT std::string build_date_time();

    // Return the pika configuration information
    PIKA_EXPORT std::string configuration_string();
}    // namespace pika::local
