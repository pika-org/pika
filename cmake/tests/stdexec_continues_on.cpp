//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <stdexec/execution.hpp>

int main()
{
    // Earlier versions of stdexec call continues_on continue_on. If stdexec has continues_on we do
    // nothing special in pika. If stdexec doesn't have continues_on, we assume it has continue_on
    // and create an alias from continue_on to continues_on. This test serves to check if continues_on is defined.
    using stdexec::continues_on;
}
