////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

namespace pika::execution::detail {
    /// TODO: implement, this is currently just a dummy
    struct resource_base
    {
        virtual ~resource_base() = default;
    };
}    // namespace pika::execution::detail
