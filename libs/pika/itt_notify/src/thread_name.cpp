//  Copyright (c) 2019 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

PIKA_GLOBAL_MODULE_FRAGMENT

#include <pika/config.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/itt_notify/thread_name.hpp>

#include <string>
#endif

#if defined(PIKA_HAVE_MODULE)
module pika.itt_notify;
#endif

namespace pika::detail {
    std::string& thread_name()
    {
        static thread_local std::string thread_name_;
        return thread_name_;
    }
}    // namespace pika::detail
