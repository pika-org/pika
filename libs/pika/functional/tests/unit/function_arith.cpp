//  Taken from the Boost.Function library

// Copyright (C) 2001-2003 Douglas Gregor
//  Copyright 2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

// For more information, see http://www.boost.org/

#include <pika/functional/function.hpp>
#include <pika/modules/testing.hpp>

double mul_ints(int x, int y)
{
    return ((double) x) * y;
}

struct int_div
{
    double operator()(int x, int y) const
    {
        return ((double) x) / y;
    };
};

int main()
{
    pika::util::function_nonser<double(int x, int y)> f;

    f = int_div();
    PIKA_TEST(f);
    PIKA_TEST_EQ(f(5, 3), 5. / 3);

    f = nullptr;
    PIKA_TEST(!f);

    f = &mul_ints;
    PIKA_TEST(f);
    PIKA_TEST_EQ(f(5, 3), 5. * 3);

    return pika::util::report_errors();
}
