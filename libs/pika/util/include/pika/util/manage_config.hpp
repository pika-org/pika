//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/string_util/from_string.hpp>

#include <map>
#include <string>
#include <vector>
#endif

#include <pika/config/warnings_prefix.hpp>

namespace pika::detail {
    struct PIKA_EXPORT manage_config
    {
        using map_type = std::map<std::string, std::string>;

        manage_config(std::vector<std::string> const& cfg);

        void add(std::vector<std::string> const& cfg);

        template <typename T>
        T get_value(std::string const& key, T dflt = T()) const
        {
            map_type::const_iterator it = config_.find(key);
            if (it != config_.end()) return pika::detail::from_string<T>((*it).second, dflt);
            return dflt;
        }

        map_type config_;
    };
}    // namespace pika::detail

#include <pika/config/warnings_suffix.hpp>
