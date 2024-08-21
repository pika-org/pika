//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_STACKTRACES)
# include <pika/debugging/backtrace/backtrace.hpp>

#if !defined(PIKA_HAVE_MODULE)
# include <cstddef>
# include <string>
#endif
#else

namespace pika::debug::detail {
    class backtrace
    {
    };

    inline std::string trace(std::size_t frames_no = PIKA_HAVE_THREAD_BACKTRACE_DEPTH)
    {
        return "";
    }
}    // namespace pika::debug::detail

#endif
