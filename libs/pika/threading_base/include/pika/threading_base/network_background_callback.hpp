//  Copyright (c) 2019 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/functional/function.hpp>

#include <cstddef>
#include <cstdint>

namespace pika::threads::detail {
    using network_background_callback_type = util::detail::function<bool(std::size_t)>;
}    // namespace pika::threads::detail
