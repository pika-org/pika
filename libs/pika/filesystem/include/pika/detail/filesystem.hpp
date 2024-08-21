//  Copyright (c) 2019-2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// This file provides a few additional filesystem utilities on top of those
/// provided by <filesystem>.

#pragma once

#include <pika/config.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <filesystem>
#include <string>
#include <system_error>
#endif

namespace pika::detail::filesystem {
    PIKA_EXPORT std::filesystem::path initial_path();
    PIKA_EXPORT std::string basename(std::filesystem::path const& p);
    PIKA_EXPORT std::filesystem::path canonical(
        std::filesystem::path const& p, std::filesystem::path const& base);
    PIKA_EXPORT std::filesystem::path canonical(
        std::filesystem::path const& p, std::filesystem::path const& base, std::error_code& ec);
}    // namespace pika::detail::filesystem
