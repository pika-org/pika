//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#if defined(PIKA_HAVE_STDEXEC)
# include <stdexec/execution.hpp>

namespace pika::execution::experimental {
    using namespace stdexec;

# if !defined(PIKA_HAVE_STDEXEC_CONTINUES_ON)
    using continues_on_t = stdexec::continue_on_t;
    inline constexpr continues_on_t continues_on{};
# endif
}    // namespace pika::execution::experimental
#endif
