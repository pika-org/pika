// defaults.cpp

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details

#include <pika/config.hpp>
#include <pika/logging/format/formatters.hpp>

#include <fmt/ostream.h>
#include <fmt/printf.h>

#include <cstdint>
#include <memory>
#include <ostream>

namespace pika::util::logging::formatter {

    idx::~idx() = default;

    struct idx_impl : idx
    {
        idx_impl()
          : value(0ull)
        {
        }

        void operator()(std::ostream& to) const override
        {
            fmt::print(to, "{:016x}", ++value);
        }

    private:
        mutable std::uint64_t value;
    };

    std::unique_ptr<idx> idx::make()
    {
        return std::unique_ptr<idx>(new idx_impl());
    }

}    // namespace pika::util::logging::formatter
