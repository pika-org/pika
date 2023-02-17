/*=============================================================================
    Copyright (c) 2014 Anton Bikineev

//  SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/
#pragma once

#include <pika/config.hpp>
#include <pika/string_util/from_string.hpp>

#include <string>
#include <type_traits>

namespace pika::detail {

    template <typename DestType, typename Config,
        std::enable_if_t<!std::is_same<DestType, std::string>::value, bool> = false>
    DestType get_entry_as(Config const& config, std::string const& key, DestType const& dflt)
    {
        std::string const& entry = config.get_entry(key, "");
        if (entry.empty())
            return dflt;
        return detail::from_string<DestType>(entry, dflt);
    }

    template <typename DestType, typename Config,
        std::enable_if_t<std::is_same<DestType, std::string>::value, bool> = false>
    DestType get_entry_as(Config const& config, std::string const& key, DestType const& dflt)
    {
        return config.get_entry(key, dflt);
    }

}    // namespace pika::detail
