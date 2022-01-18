//  Copyright (c) 2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/iterator_support/tests/iter_sent.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/util/ranges_facilities.hpp>

#include <cstdint>
#include <vector>

void test_ranges_next()
{
    std::vector<std::int16_t> v = {1, 5, 3, 6};
    auto it = v.begin();

    auto next1 = pika::ranges::next(it);
    PIKA_TEST_EQ(*next1, 5);

    auto next2 = pika::ranges::next(it, 2);
    PIKA_TEST_EQ(*next2, 3);

    auto next3 = pika::ranges::next(it, sentinel<std::int16_t>(3));
    PIKA_TEST_EQ(*next3, 3);

    auto next4 = pika::ranges::next(it, 2, v.end());
    PIKA_TEST_EQ(*next4, 3);

    auto next5 = pika::ranges::next(it, 42, v.end());
    PIKA_TEST(next5 == v.end());
}

int main()
{
    test_ranges_next();
    return pika::util::report_errors();
}
