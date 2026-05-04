//  Copyright (c) 2026 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <exec/completion_signatures.hpp>
#include <stdexec/execution.hpp>

int main()
{
    // Newer versions of stdexec deprecate transform_completion_signatures_of and provide
    // ::experimental::execution::transform_completion_signatures in
    // <exec/completion_signatures.hpp> as the public replacement.
    using type = decltype(::experimental::execution::transform_completion_signatures(
        stdexec::get_completion_signatures<decltype(stdexec::just()), stdexec::env<>>(),
        ::experimental::execution::keep_completion<stdexec::set_value_t>{},
        ::experimental::execution::keep_completion<stdexec::set_error_t>{},
        ::experimental::execution::keep_completion<stdexec::set_stopped_t>{},
        stdexec::completion_signatures<>{}));
}
