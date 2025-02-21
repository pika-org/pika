//  Copyright (c) 2024 ETH Zurich
//  Copyright (c) 2006, Giovanni P. Deretta
//  Copyright (c) 2007 Robert Perricone
//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2013-2016 Thomas Heller
//  Copyright (c) 2017 Christopher Taylor
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

// The concept is inspired by the following links:
//
// https://rethinkdb.com/blog/handling-stack-overflow-on-custom-stacks/
// http://www.evanjones.ca/software/threading.html

#pragma once

#include <pika/config.hpp>

#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)

# include <signal.h>

namespace pika::threads::coroutines::detail {
    PIKA_EXPORT void set_stackoverflow_detection(bool);
    PIKA_EXPORT bool get_stackoverflow_detection();
    PIKA_EXPORT void set_sigsegv_handler(struct sigaction& action, stack_t& segv_stack);
}    // namespace pika::threads::coroutines::detail
#endif
