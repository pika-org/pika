//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if !defined(PIKA_HAVE_MODULE)
#include <utility>
#endif

namespace pika::detail {
    template <typename F, typename... Args>
    using invoke_result_plain_function_t = decltype(std::declval<F>()(std::declval<Args>()...));
}
