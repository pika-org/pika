//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/datastructures/optional.hpp>
#include <pika/modules/testing.hpp>

int main()
{
    pika::util::optional<int> x;
    int y = 42;
    x = y;

    PIKA_TEST(x.has_value());
    PIKA_TEST_EQ(*x, 42);

    return pika::util::report_errors();
}
