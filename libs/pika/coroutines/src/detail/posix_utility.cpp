//  Copyright (c) 2005-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

PIKA_GLOBAL_MODULE_FRAGMENT

#include <pika/config.hpp>
#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__) ||            \
    defined(__APPLE__)

#if !defined(PIKA_HAVE_MODULE)
# include <pika/coroutines/detail/posix_utility.hpp>
#endif

#if defined(PIKA_HAVE_MODULE)
module pika.coroutines;
#endif

namespace pika::threads::coroutines::detail ::posix {
    ///////////////////////////////////////////////////////////////////////
    // this global (urghhh) variable is used to control whether guard pages
    // will be used or not
    PIKA_EXPORT bool use_guard_pages = true;
}    // namespace pika::threads::coroutines::detail::posix

#else

#if defined(PIKA_HAVE_MODULE)
module pika.coroutines;
#endif

#endif
