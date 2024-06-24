//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if !defined(PIKA_HAVE_MODULE)

#include <pika/config.hpp>
#include <pika/runtime_configuration/runtime_configuration.hpp>

#include <string>

namespace pika::detail {
    PIKA_EXPORT void init_logging(pika::util::runtime_configuration& ini);
}    // namespace pika::detail

#endif
