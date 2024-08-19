//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2012 Hartmut Kaiser
//  Copyright (c) 2009 Oliver Kowalke
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

PIKA_GLOBAL_MODULE_FRAGMENT

#include <pika/config.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/coroutines/detail/context_base.hpp>
#include <pika/coroutines/detail/coroutine_impl.hpp>
#endif

#if defined(PIKA_HAVE_MODULE)
module pika.coroutines;
#endif

namespace pika::threads::coroutines::detail {
    template class context_base<coroutine_impl>;
}    // namespace pika::threads::coroutines::detail
