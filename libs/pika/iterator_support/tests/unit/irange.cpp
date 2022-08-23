//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/iterator_support/irange.hpp>
#include <pika/iterator_support/range.hpp>
#include <pika/testing.hpp>

#include <vector>

template <typename Range, typename T>
void test(Range&& rng, T&& ref)
{
    auto rb = std::begin(ref);
    for (auto r : rng)
    {
        PIKA_TEST(rb != std::end(ref));
        PIKA_TEST_EQ(r, *rb++);
    }
    PIKA_TEST(rb == std::end(ref));
    PIKA_TEST_EQ(std::distance(std::begin(rng), std::end(rng)),
        std::distance(std::begin(ref), std::end(ref)));
}

template <typename Range>
void test_unit_stride()
{
    test(Range(0, 0), std::vector<int>{});
    test(Range(-10, -10), std::vector<int>{});
    test(Range(14, 14), std::vector<int>{});
    test(Range(0, 1), std::vector<int>{0});
    test(Range(9, 10), std::vector<int>{9});
    test(Range(-11, -10), std::vector<int>{-11});
    test(Range(0, 4), std::vector<int>{0, 1, 2, 3});
    test(Range(-10, -8), std::vector<int>{-10, -9});
    test(Range(10, 13), std::vector<int>{10, 11, 12});
    test(Range(10, 13), std::vector<int>{10, 11, 12});
}

void test_non_unit_stride()
{
    test(pika::detail::strided_irange(0, 0, 1), std::vector<int>{});
    test(pika::detail::strided_irange(-10, -10, 1), std::vector<int>{});
    test(pika::detail::strided_irange(14, 14, 1), std::vector<int>{});
    test(pika::detail::strided_irange(0, 1, 1), std::vector<int>{0});
    test(pika::detail::strided_irange(9, 10, 1), std::vector<int>{9});
    test(pika::detail::strided_irange(-11, -10, 1), std::vector<int>{-11});
    test(pika::detail::strided_irange(0, 4, 1), std::vector<int>{0, 1, 2, 3});
    test(pika::detail::strided_irange(-10, -8, 1), std::vector<int>{-10, -9});
    test(pika::detail::strided_irange(10, 13, 1), std::vector<int>{10, 11, 12});
    test(pika::detail::strided_irange(0, 0, 3), std::vector<int>{});
    test(pika::detail::strided_irange(-10, -10, 3), std::vector<int>{});
    test(pika::detail::strided_irange(14, 14, 3), std::vector<int>{});
    test(pika::detail::strided_irange(0, 0, -3), std::vector<int>{});
    test(pika::detail::strided_irange(-10, -10, -3), std::vector<int>{});
    test(pika::detail::strided_irange(14, 14, -3), std::vector<int>{});
    test(pika::detail::strided_irange(0, 1, 3), std::vector<int>{0});
    test(pika::detail::strided_irange(9, 10, 3), std::vector<int>{9});
    test(pika::detail::strided_irange(-11, -10, 3), std::vector<int>{-11});
    test(pika::detail::strided_irange(0, 4, 3), std::vector<int>{0, 3});
    test(pika::detail::strided_irange(-10, -8, 3), std::vector<int>{-10});
    test(pika::detail::strided_irange(10, 12, 3), std::vector<int>{10});
    test(pika::detail::strided_irange(10, 13, 3), std::vector<int>{10});
    test(pika::detail::strided_irange(10, 14, 3), std::vector<int>{10, 13});
    test(pika::detail::strided_irange(10, 15, 3), std::vector<int>{10, 13});
    test(pika::detail::strided_irange(-10, -12, -3), std::vector<int>{-10});
    test(pika::detail::strided_irange(-10, -13, -3), std::vector<int>{-10});
    test(
        pika::detail::strided_irange(-10, -14, -3), std::vector<int>{-10, -13});
    test(
        pika::detail::strided_irange(-10, -15, -3), std::vector<int>{-10, -13});
}

int main()
{
    test_unit_stride<pika::detail::irange<int>>();
    test_unit_stride<pika::detail::strided_irange<int, int>>();
    test_non_unit_stride();

    return 0;
}
