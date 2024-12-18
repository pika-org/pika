//  Taken from the Boost.Function library

//  Copyright Douglas Gregor 2008.
//  Copyright 2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
// Boost.Function library

// For more information, see http://www.boost.org

#include <pika/functional/function.hpp>
#include <pika/testing.hpp>

struct tried_to_copy
{
};

struct MaybeThrowOnCopy
{
    MaybeThrowOnCopy(int value = 0)
      : value(value)
    {
    }

    MaybeThrowOnCopy(MaybeThrowOnCopy const& other)
      : value(other.value)
    {
        if (throwOnCopy) throw tried_to_copy();
    }

    // NOLINTNEXTLINE(bugprone-unhandled-self-assignment)
    MaybeThrowOnCopy& operator=(MaybeThrowOnCopy const& other)
    {
        if (throwOnCopy) throw tried_to_copy();
        value = other.value;
        return *this;
    }

    int operator()() { return value; }

    int value;

    // Make sure that this function object doesn't trigger the
    // small-object optimization in Function.
    float padding[100];

    static bool throwOnCopy;
};

bool MaybeThrowOnCopy::throwOnCopy = false;

int main(int, char*[])
{
    pika::util::detail::function<int()> f;
    pika::util::detail::function<int()> g;

    MaybeThrowOnCopy::throwOnCopy = false;
    f = MaybeThrowOnCopy(1);
    g = MaybeThrowOnCopy(2);
    PIKA_TEST_EQ(f(), 1);
    PIKA_TEST_EQ(g(), 2);

    MaybeThrowOnCopy::throwOnCopy = true;
    f.swap(g);
    PIKA_TEST_EQ(f(), 2);
    PIKA_TEST_EQ(g(), 1);

    return 0;
}
