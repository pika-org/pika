//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

PIKA_GLOBAL_MODULE_FRAGMENT

#include <pika/config.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/topology/cpu_mask.hpp>

#include <iomanip>
#include <sstream>
#include <string>
#endif

#if defined(PIKA_HAVE_MODULE)
module pika.topology;
#endif

// clang-format off
#if !defined(PIKA_HAVE_MORE_THAN_64_THREADS) ||                                 \
    (defined(PIKA_HAVE_MAX_CPU_COUNT) && PIKA_HAVE_MAX_CPU_COUNT <= 64)

#define PIKA_CPU_MASK_PREFIX "0x"

#else

#  if defined(PIKA_HAVE_MAX_CPU_COUNT)
#    define PIKA_CPU_MASK_PREFIX "0b"
#  else
#    define PIKA_CPU_MASK_PREFIX "0x"
#  endif

#endif
// clang-format on

namespace pika::threads::detail {
    std::string to_string(mask_cref_type val)
    {
        std::ostringstream ostr;
        // TODO: invalid operands to binary expression?
        // ostr << std::hex << PIKA_CPU_MASK_PREFIX << val;
        return ostr.str();
    }
}    // namespace pika::threads::detail
