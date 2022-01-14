//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2017      Denis Blank
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/debugging/attach_debugger.hpp>

#include <iostream>

#if defined(PIKA_HAVE_UNISTD_H)
#include <unistd.h>
#endif

#if defined(PIKA_WINDOWS)
#include <Windows.h>
#endif    // PIKA_WINDOWS

namespace pika { namespace util {
    void attach_debugger()
    {
#if defined(_POSIX_VERSION) && defined(PIKA_HAVE_UNISTD_H)
        volatile int i = 0;
        std::cerr << "PID: " << getpid()
                  << " ready for attaching debugger. Once attached set i = 1 "
                     "and continue"
                  << std::endl;
        while (i == 0)
        {
            sleep(1);
        }
#elif defined(PIKA_WINDOWS)
        DebugBreak();
#endif
    }
}}    // namespace pika::util
