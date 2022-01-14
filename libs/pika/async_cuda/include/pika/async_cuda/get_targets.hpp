///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <pika/local/config.hpp>
#include <pika/modules/futures.hpp>

#include <vector>

namespace pika { namespace cuda { namespace experimental {
    struct PIKA_EXPORT target;

    PIKA_EXPORT std::vector<target> get_local_targets();
    PIKA_EXPORT void print_local_targets();

}}}    // namespace pika::cuda::experimental
