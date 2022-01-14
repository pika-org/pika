//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

#include <cstddef>

#ifdef PIKA_HAVE_SPINLOCK_DEADLOCK_DETECTION
namespace pika { namespace util { namespace detail {
    PIKA_EXPORT void set_spinlock_break_on_deadlock_enabled(bool enabled);
    PIKA_EXPORT bool get_spinlock_break_on_deadlock_enabled();
    PIKA_EXPORT void set_spinlock_deadlock_detection_limit(
        std::size_t limit);
    PIKA_EXPORT std::size_t get_spinlock_deadlock_detection_limit();
}}}    // namespace pika::util::detail
#endif
