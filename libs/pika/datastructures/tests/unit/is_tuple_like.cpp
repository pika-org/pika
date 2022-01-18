//  Copyright (c) 2017 Denis Blank
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <array>
#include <utility>
#include <vector>

#include <pika/local/config.hpp>
#include <pika/datastructures/traits/is_tuple_like.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/modules/testing.hpp>

void tuple_like_true()
{
    using pika::traits::is_tuple_like;

    PIKA_TEST_EQ((is_tuple_like<pika::tuple<int, int, int>>::value), true);
    PIKA_TEST_EQ((is_tuple_like<std::pair<int, int>>::value), true);
    PIKA_TEST_EQ((is_tuple_like<std::array<int, 4>>::value), true);
}

void tuple_like_false()
{
    using pika::traits::is_tuple_like;

    PIKA_TEST_EQ((is_tuple_like<int>::value), false);
    PIKA_TEST_EQ((is_tuple_like<std::vector<int>>::value), false);
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    {
        tuple_like_true();
        tuple_like_false();
    }

    return pika::util::report_errors();
}
