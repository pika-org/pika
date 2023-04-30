//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_STDEXEC)
# include <pika/execution_base/stdexec_forward.hpp>
#else

# include <pika/execution/algorithms/execute.hpp>

namespace pika::execution::experimental {

    // A simple scheduler that executes its continuation inline, on the
    // thread of the caller of start().
    // Note that this has been reduced to the minimal code that compiles and runs
    struct inline_scheduler
    {
        template <typename F>
        void execute(F&& f) const
        {
            PIKA_INVOKE(f, );
        }

        template <typename F>
        friend void tag_invoke(execute_t, inline_scheduler const& sched, F&& f)
        {
            sched.execute(PIKA_FORWARD(F, f));
        }
    };

#endif
}    // namespace pika::execution::experimental
