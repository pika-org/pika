// Taken from the Boost.Bind library
//
//  mem_fn_dm_test.cpp - data members
//
//  Copyright (c) 2005 Peter Dimov
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

#include <pika/local/config.hpp>

#if defined(PIKA_MSVC)
#pragma warning(disable : 4786)    // identifier truncated in debug info
#pragma warning(disable : 4710)    // function not inlined
#pragma warning(                                                               \
    disable : 4711)    // function selected for automatic inline expansion
#pragma warning(disable : 4514)    // unreferenced inline removed
#endif

#include <pika/functional/mem_fn.hpp>

#include <iostream>

#include <pika/modules/testing.hpp>

struct X
{
    int m;
};

int main()
{
    X x = {0};

    pika::util::mem_fn (&X::m)(x) = 401;

    PIKA_TEST_EQ(x.m, 401);
    PIKA_TEST_EQ(pika::util::mem_fn(&X::m)(x), 401);

    pika::util::mem_fn (&X::m)(&x) = 502;

    PIKA_TEST_EQ(x.m, 502);
    PIKA_TEST_EQ(pika::util::mem_fn(&X::m)(&x), 502);

    X* px = &x;

    pika::util::mem_fn (&X::m)(px) = 603;

    PIKA_TEST_EQ(x.m, 603);
    PIKA_TEST_EQ(pika::util::mem_fn(&X::m)(px), 603);

    X const& cx = x;
    X const* pcx = &x;

    PIKA_TEST_EQ(pika::util::mem_fn(&X::m)(cx), 603);
    PIKA_TEST_EQ(pika::util::mem_fn(&X::m)(pcx), 603);

    return pika::util::report_errors();
}
