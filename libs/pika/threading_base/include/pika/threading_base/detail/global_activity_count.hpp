//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <cstddef>
#endif

namespace pika::threads::detail {
    PIKA_EXPORT void increment_global_activity_count();
    PIKA_EXPORT void decrement_global_activity_count();
    PIKA_EXPORT std::size_t get_global_activity_count();
}    // namespace pika::threads::detail
