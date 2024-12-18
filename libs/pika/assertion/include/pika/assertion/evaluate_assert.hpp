//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#include <pika/assertion/source_location.hpp>

#include <string>
#include <utility>

namespace pika::detail {
    /// \cond NOINTERNAL
    PIKA_EXPORT void handle_assert(
        source_location const& loc, char const* expr, std::string const& msg) noexcept;
    /// \endcond
}    // namespace pika::detail
