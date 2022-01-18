//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/functional/function.hpp>

namespace pika { namespace parallel { namespace util { namespace detail {
    using parallel_exception_termination_handler_type =
        pika::util::function_nonser<void()>;

    PIKA_EXPORT void set_parallel_exception_termination_handler(
        parallel_exception_termination_handler_type f);

    PIKA_NORETURN PIKA_EXPORT void parallel_exception_termination_handler();
}}}}    // namespace pika::parallel::util::detail
