//  Copyright (c) 2014 Erik Schnetter
//  Copyright (c) 2014 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/functional/invoke.hpp>
#include <pika/functional/traits/is_invocable.hpp>
#include <pika/modules/testing.hpp>

struct s
{
    int f() const
    {
        return 42;
    }
};

struct p
{
    s x;
    s const& operator*() const
    {
        return x;
    }
};

///////////////////////////////////////////////////////////////////////////////
int main()
{
    using pika::is_invocable_v;
    using pika::util::invoke;

    typedef int (s::*mem_fun_ptr)();
    PIKA_TEST_MSG((is_invocable_v<mem_fun_ptr, p> == false), "mem-fun-ptr");

    typedef int (s::*const_mem_fun_ptr)() const;
    PIKA_TEST_MSG(
        (is_invocable_v<const_mem_fun_ptr, p> == true), "const-mem-fun-ptr");

    PIKA_TEST_EQ(invoke(&s::f, p()), 42);

    return pika::util::report_errors();
}
