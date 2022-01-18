//  Copyright (c) 2005-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#if defined(__linux) || defined(linux) || defined(__linux__) ||                \
    defined(__FreeBSD__) || defined(__APPLE__)
#include <pika/coroutines/detail/posix_utility.hpp>

namespace pika { namespace threads { namespace coroutines { namespace detail {
    namespace posix {
        ///////////////////////////////////////////////////////////////////////
        // this global (urghhh) variable is used to control whether guard pages
        // will be used or not
        PIKA_EXPORT bool use_guard_pages = true;
}}}}}    // namespace pika::threads::coroutines::detail::posix
#endif
