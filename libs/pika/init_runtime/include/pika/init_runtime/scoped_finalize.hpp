//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/init_runtime/init_runtime.hpp>

namespace pika {
    struct PIKA_NODISCARD scoped_finalize
    {
        scoped_finalize() = default;
        PIKA_EXPORT ~scoped_finalize();
    };
}    // namespace pika
