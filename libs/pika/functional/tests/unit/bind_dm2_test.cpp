//  Taken from the Boost.Bind library
//
//  bind_dm2_test.cpp - data members, advanced uses
//
//  Copyright (c) 2005 Peter Dimov
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

#if defined(PIKA_MSVC)
#pragma warning(disable : 4786)    // identifier truncated in debug info
#pragma warning(disable : 4710)    // function not inlined
#pragma warning(                                                               \
    disable : 4711)    // function selected for automatic inline expansion
#pragma warning(disable : 4514)    // unreferenced inline removed
#endif

#include <pika/functional/bind.hpp>

namespace placeholders = pika::util::placeholders;

#include <functional>
#include <iostream>
#include <string>

#include <pika/modules/testing.hpp>

struct X
{
    int m;
};

struct Y
{
    char m[64];
};

int main()
{
    X x = {0};
    X* px = &x;

    pika::util::bind(&X::m, placeholders::_1)(px) = 42;

    PIKA_TEST_EQ(x.m, 42);

    pika::util::bind(&X::m, std::ref(x))() = 17041;

    PIKA_TEST_EQ(x.m, 17041);

    X const* pcx = &x;

    PIKA_TEST_EQ(pika::util::bind(&X::m, placeholders::_1)(pcx), 17041L);
    PIKA_TEST_EQ(pika::util::bind(&X::m, pcx)(), 17041L);

    Y y = {"test"};
    std::string v("test");

    PIKA_TEST_EQ(pika::util::bind(&Y::m, &y)(), v);
    PIKA_TEST_EQ(pika::util::bind(&Y::m, &y)(), v);

    return pika::util::report_errors();
}
