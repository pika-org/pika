//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <stdexec/execution.hpp>

int main()
{
    // Newer versions of stdexec deprecate transform_completion_signatures_of and provide
    // __transform_completion_signatures_of_t as a non-deprecated replacement.
    using type =
        stdexec::__transform_completion_signatures_of_t<decltype(stdexec::just()), stdexec::env<>>;
}
