//  Taken from the Boost.Function library

//  Copyright Douglas Gregor 2002-2003.
//  Copyright 2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

// For more information, see http://www.boost.org

#include <pika/functional/bind.hpp>
#include <pika/functional/function.hpp>
#include <pika/modules/testing.hpp>

#include <cstdlib>

static unsigned func_impl(int arg1, bool arg2, double arg3)
{
    using namespace std;
    return abs(static_cast<int>((arg2 ? arg1 : 2 * arg1) * arg3));
}

int main(int, char*[])
{
    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    pika::util::function_nonser<unsigned(bool, double)> f1 =
        pika::util::bind(func_impl, 15, _1, _2);
    pika::util::function_nonser<unsigned(double)> f2 =
        pika::util::bind(f1, false, _1);
    pika::util::function_nonser<unsigned()> f3 = pika::util::bind(f2, 4.0);

    PIKA_TEST_EQ(f3(), 120u);

    return pika::util::report_errors();
}
