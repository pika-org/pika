//  Copyright (c) 2016 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/iterator_support/range.hpp>
#include <pika/modules/testing.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
void array_range()
{
    int r[3] = {0, 1, 2};
    PIKA_TEST(pika::util::begin(r) == &r[0]);
    PIKA_TEST(pika::util::end(r) == &r[3]);

    int const cr[3] = {0, 1, 2};
    PIKA_TEST(pika::util::begin(cr) == &cr[0]);
    PIKA_TEST(pika::util::end(cr) == &cr[3]);
    PIKA_TEST_EQ(pika::util::size(cr), 3u);
    PIKA_TEST_EQ(pika::util::empty(cr), false);
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
    member r = member();
    PIKA_TEST(pika::util::begin(r) == &r.x);
    PIKA_TEST(pika::util::end(r) == &r.x + 1);

    member const cr = member();
    PIKA_TEST(pika::util::begin(cr) == &cr.x);
    PIKA_TEST(pika::util::end(cr) == &cr.x + 1);
    PIKA_TEST_EQ(pika::util::size(cr), 1u);
    PIKA_TEST_EQ(pika::util::empty(cr), false);
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
    adl::free r = adl::free();
    PIKA_TEST(pika::util::begin(r) == &r.x);
    PIKA_TEST(pika::util::end(r) == &r.x + 1);

    adl::free const cr = adl::free();
    PIKA_TEST(pika::util::begin(cr) == &cr.x);
    PIKA_TEST(pika::util::end(cr) == &cr.x + 1);
    PIKA_TEST_EQ(pika::util::size(cr), 1u);
    PIKA_TEST_EQ(pika::util::empty(cr), false);
}

///////////////////////////////////////////////////////////////////////////////
void vector_range()
{
    std::vector<int> r(3);
    PIKA_TEST(pika::util::begin(r) == r.begin());
    PIKA_TEST(pika::util::end(r) == r.end());

    std::vector<int> cr(3);
    PIKA_TEST(pika::util::begin(cr) == cr.begin());
    PIKA_TEST(pika::util::end(cr) == cr.end());
    PIKA_TEST_EQ(pika::util::size(cr), 3u);
    PIKA_TEST_EQ(pika::util::empty(cr), false);
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

    return pika::util::report_errors();
}
