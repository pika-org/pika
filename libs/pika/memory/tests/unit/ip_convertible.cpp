//  wp_convertible_test.cpp
//
//  Copyright (c) 2008 Peter Dimov
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt

#include <pika/local/config.hpp>
#include <pika/modules/memory.hpp>
#include <pika/modules/testing.hpp>

//
struct W
{
};

void intrusive_ptr_add_ref(W*) {}

void intrusive_ptr_release(W*) {}

struct X : public virtual W
{
};

struct Y : public virtual W
{
};

struct Z : public X
{
};

int f(pika::intrusive_ptr<X>)
{
    return 1;
}

int f(pika::intrusive_ptr<Y>)
{
    return 2;
}

int main()
{
    PIKA_TEST_EQ(1, f(pika::intrusive_ptr<Z>()));
    return pika::util::report_errors();
}
