////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Thomas Heller
//  Copyright (c) 2008 Peter Dimov
//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <pika/local/config.hpp>
#include <pika/execution_base/detail/spinlock_deadlock_detection.hpp>

#include <cstddef>

#ifdef PIKA_HAVE_SPINLOCK_DEADLOCK_DETECTION
namespace pika { namespace util { namespace detail {
    static bool spinlock_break_on_deadlock_enabled = false;
    static std::size_t spinlock_deadlock_detection_limit =
        PIKA_SPINLOCK_DEADLOCK_DETECTION_LIMIT;

    void set_spinlock_break_on_deadlock_enabled(bool enabled)
    {
        spinlock_break_on_deadlock_enabled = enabled;
    }

    bool get_spinlock_break_on_deadlock_enabled()
    {
        return spinlock_break_on_deadlock_enabled;
    }

    void set_spinlock_deadlock_detection_limit(std::size_t limit)
    {
        spinlock_deadlock_detection_limit = limit;
    }

    std::size_t get_spinlock_deadlock_detection_limit()
    {
        return spinlock_deadlock_detection_limit;
    }
}}}    // namespace pika::util::detail
#endif
