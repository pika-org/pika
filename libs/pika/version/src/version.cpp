////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <pika/config.hpp>
#include <pika/config/config_strings.hpp>
#include <pika/config/version.hpp>
#include <pika/modules/format.hpp>
#include <pika/prefix/find_prefix.hpp>
#include <pika/preprocessor/stringize.hpp>
#include <pika/version.hpp>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace pika {
    std::uint8_t major_version()
    {
        return PIKA_VERSION_MAJOR;
    }

    std::uint8_t minor_version()
    {
        return PIKA_VERSION_MINOR;
    }

    std::uint8_t patch_version()
    {
        return PIKA_VERSION_PATCH;
    }

    std::uint32_t full_version()
    {
        return PIKA_VERSION_FULL;
    }

    std::string full_version_as_string()
    {
        return pika::util::format("{}.{}.{}",    //-V609
            PIKA_VERSION_MAJOR, PIKA_VERSION_MINOR, PIKA_VERSION_PATCH);
    }

    std::string tag()
    {
        return PIKA_VERSION_TAG;
    }

    std::string copyright()
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

    // Returns the pika full build information string.
    std::string full_build_string()
    {
        std::ostringstream strm;
        strm << "{config}:\n"
             << configuration_string() << "{version}: " << build_string()
             << "\n"
             << "{build-type}: " << build_type() << "\n"
             << "{date}: " << build_date_time() << "\n";

        return strm.str();
    }

    ///////////////////////////////////////////////////////////////////////////
    std::string configuration_string()
    {
        std::ostringstream strm;

        strm << "pika:\n";

        char const* const* p = pika::config_strings;
        while (*p)
            strm << "  " << *p++ << "\n";
        strm << "\n";

        return strm.str();
    }

    std::string build_string()
    {
        return pika::util::format("V{}{}, Git: {:.10}",    //-V609
            full_version_as_string(), PIKA_VERSION_TAG, PIKA_HAVE_GIT_COMMIT);
    }

    std::string complete_version()
    {
        std::string version = pika::util::format("Version:\n"
                                                 "  pika: {}\n"
                                                 "\n"
                                                 "Build:\n"
                                                 "  Type: {}\n"
                                                 "  Date: {}\n",
            build_string(), build_type(), build_date_time());

        return version;
    }

    std::string build_type()
    {
        return PIKA_PP_STRINGIZE(PIKA_BUILD_TYPE);
    }

    std::string build_date_time()
    {
        return std::string(__DATE__) + " " + __TIME__;
    }
}    // namespace pika
