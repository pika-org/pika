//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#if defined(PIKA_HAVE_STDEXEC)
# include <stdexec/execution.hpp>
# if defined(PIKA_HAVE_STDEXEC_EXEC_EXECUTE)
#  include <exec/execute.hpp>
#  include <exec/split.hpp>
# endif

namespace pika::execution::experimental {
    using namespace stdexec;
    using stdexec::get_completion_scheduler_t;
# if defined(PIKA_HAVE_STDEXEC_EXEC_EXECUTE)
    // https://github.com/NVIDIA/stdexec/pull/1830 and #1861 move execute and split to
    // experimental::execution namespace.
    using ::experimental::execution::split;
    using ::experimental::execution::split_t;
    // execute and execute_t are deprecated in exec/execute.hpp, use __execute/__execute_t
    // FIXME: remove those 2 when we update to a new stdexec
    using execute_t = ::experimental::execution::__execute_t;
    inline constexpr execute_t const& execute = ::experimental::execution::__execute;
# endif

    // continue_on_t has been renamed to continues_on_t (valid in version from 02.2026)
# if !defined(PIKA_HAVE_STDEXEC_CONTINUES_ON)
    using continues_on_t = stdexec::continue_on_t;
    inline constexpr continues_on_t continues_on{};
# endif

    // empty_env is now deprecated in stdexec
# if defined(PIKA_HAVE_STDEXEC_ENV)
    using empty_env = stdexec::env<>;
# endif
}    // namespace pika::execution::experimental
#endif
