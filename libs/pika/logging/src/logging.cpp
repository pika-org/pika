//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>

#if defined(PIKA_HAVE_LOGGING)
#include <pika/modules/filesystem.hpp>
#include <pika/modules/logging.hpp>
#include <pika/util/from_string.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace util {
    PIKA_DEFINE_LOG(app, disable_all)
    PIKA_DEFINE_LOG(app_console, disable_all)
    PIKA_DEFINE_LOG(app_error, fatal)
    PIKA_DEFINE_LOG(debuglog, disable_all)
    PIKA_DEFINE_LOG(debuglog_console, disable_all)
    PIKA_DEFINE_LOG(debuglog_error, fatal)
    PIKA_DEFINE_LOG(pika, disable_all)
    PIKA_DEFINE_LOG(pika_console, disable_all)
    PIKA_DEFINE_LOG(pika_error, fatal)
    PIKA_DEFINE_LOG(timing, disable_all)
    PIKA_DEFINE_LOG(timing_console, disable_all)

    namespace detail {
        pika::util::logging::level get_log_level(
            std::string const& env, bool allow_always)
        {
            try
            {
                int env_val = pika::util::from_string<int>(env);
                if (env_val < 0)
                    return pika::util::logging::level::disable_all;

                switch (env_val)
                {
                case 0:
                    return allow_always ?
                        pika::util::logging::level::always :
                        pika::util::logging::level::disable_all;
                case 1:
                    return pika::util::logging::level::fatal;
                case 2:
                    return pika::util::logging::level::error;
                case 3:
                    return pika::util::logging::level::warning;
                case 4:
                    return pika::util::logging::level::info;
                default:
                    break;
                }
                return pika::util::logging::level::debug;
            }
            catch (pika::util::bad_lexical_cast const&)
            {
                return pika::util::logging::level::disable_all;
            }
        }
    }    // namespace detail
}}       // namespace pika::util

///////////////////////////////////////////////////////////////////////////////
#include <pika/logging/detail/logger.hpp>

namespace pika { namespace util { namespace logging {

    void logger::turn_cache_off()
    {
        if (m_is_caching_off)
            return;    // already turned off

        m_is_caching_off = true;

        // dump messages
        std::vector<message> msgs;
        std::swap(m_cache, msgs);

        for (auto& msg : msgs)
            m_writer(msg);
    }

}}}    // namespace pika::util::logging

#endif    // PIKA_HAVE_LOGGING
