//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/testing.hpp>

#include <cstdlib>
#include <exception>
#include <iostream>

namespace ex = pika::execution::experimental;

int main()
{
    std::set_terminate([] {
        std::cout << "std::terminate called\n";
        std::exit(pika::detail::report_errors());
    });

    {
        PIKA_TEST(true);
        auto rs = ex::require_started(ex::just());
    }

    // The test should terminate above because rs is never connected or started
    PIKA_TEST(false);
}
