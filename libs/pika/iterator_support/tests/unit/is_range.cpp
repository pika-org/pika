//  Copyright (c) 2016 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/iterator_support/traits/is_range.hpp>
#include <pika/testing.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
void array_range()
{
    using range = int[3];

    PIKA_TEST_MSG((pika::traits::is_range<range>::value == true), "array");
    PIKA_TEST_MSG((pika::traits::is_range<range const>::value == true), "array-const");
}

///////////////////////////////////////////////////////////////////////////////
struct member
{
    int x;

    int* begin()
    {
        return &x;
    }

    int const* begin() const
    {
        return &x;
    }

    int* end()
    {
        return &x + 1;
    }

    int const* end() const
    {
        return &x + 1;
    }
};

void member_range()
{
    using range = member;

    PIKA_TEST_MSG((pika::traits::is_range<range>::value == true), "member-const");
    PIKA_TEST_MSG((pika::traits::is_range<range const>::value == true), "member-const");
}

///////////////////////////////////////////////////////////////////////////////
namespace adl {
    struct free
    {
        int x;
    };

    int* begin(free& r)
    {
        return &r.x;
    }

    int const* begin(free const& r)
    {
        return &r.x;
    }

    int* end(free& r)
    {
        return &r.x + 1;
    }

    int const* end(free const& r)
    {
        return &r.x + 1;
    }
}    // namespace adl

void adl_range()
{
    using range = adl::free;

    PIKA_TEST_MSG((pika::traits::is_range<range>::value == true), "adl-const");
    PIKA_TEST_MSG((pika::traits::is_range<range const>::value == true), "adl-const");
}

///////////////////////////////////////////////////////////////////////////////
void vector_range()
{
    using range = std::vector<int>;

    PIKA_TEST_MSG((pika::traits::is_range<range>::value == true), "vector");
    PIKA_TEST_MSG((pika::traits::is_range<range const>::value == true), "vector-const");
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    {
        array_range();
        member_range();
        adl_range();
        vector_range();
    }

    return 0;
}
