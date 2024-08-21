//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/execution_base/resource_base.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/timing/steady_clock.hpp>

#include <cstdint>
#endif

namespace pika::execution::detail {
    struct context_base
    {
        virtual ~context_base() = default;

        virtual resource_base const& resource() const = 0;
    };
}    // namespace pika::execution::detail
