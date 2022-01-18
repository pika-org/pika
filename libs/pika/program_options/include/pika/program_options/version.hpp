// Copyright Vladimir Prus 2004.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/program_options/config.hpp>

/** The version of the source interface.
    The value will be incremented whenever a change is made which might
    cause compilation errors for existing code.
*/
#ifdef PIKA_PROGRAM_OPTIONS_VERSION
#error PIKA_PROGRAM_OPTIONS_VERSION already defined
#endif
#define PIKA_PROGRAM_OPTIONS_VERSION 2

// Signal that implicit options will use values from next
// token, if available.
#define PIKA_PROGRAM_OPTIONS_IMPLICIT_VALUE_NEXT_TOKEN 1
