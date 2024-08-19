//  Copyright (c) 2008-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

PIKA_GLOBAL_MODULE_FRAGMENT

#include <pika/config.hpp>
#include <pika/assert.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/coroutines/detail/coroutine_self.hpp>
#endif

#include <cstddef>

#if defined(PIKA_HAVE_MODULE)
module pika.coroutines;
#endif

namespace pika::threads::coroutines::detail {
    coroutine_self*& coroutine_self::local_self()
    {
        static thread_local coroutine_self* local_self_ = nullptr;
        return local_self_;
    }
}    // namespace pika::threads::coroutines::detail
