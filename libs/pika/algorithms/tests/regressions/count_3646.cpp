//  Copyright (c) 2019 Piotr Mikolajczyk
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// #3646: Parallel algorithms should accept iterator/sentinel pairs

#include <pika/iterator_support/tests/iter_sent.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/container_algorithms/count.hpp>

#include <cstddef>
#include <cstdint>
#include <iterator>

struct bit_counting_iterator : public iterator<std::int64_t>
{
    using difference_type = std::ptrdiff_t;
    using value_type = std::int64_t;
    using iterator_category = std::forward_iterator_tag;
    using pointer = std::int64_t const*;
    using reference = std::int64_t const&;

    explicit bit_counting_iterator(int64_t initialState)
      : iterator<int64_t>(initialState)
    {
    }

private:
    std::int64_t countBits(std::int64_t v) const
    {
        int counter = 0;
        while (v != 0)
        {
            if (v & 1)
                ++counter;
            v >>= 1;
        }
        return counter;
    }
};

void test_count()
{
    using Iter = bit_counting_iterator;
    using Sent = sentinel<std::int64_t>;

    auto stdResult = std::count(Iter{0}, Iter{33}, std::int64_t{1});

    auto result = pika::ranges::count(
        pika::execution::seq, Iter{0}, Sent{33}, std::int64_t{1});

    PIKA_TEST_EQ(result, stdResult);

    result = pika::ranges::count(
        pika::execution::par, Iter{0}, Sent{33}, std::int64_t{1});

    PIKA_TEST_EQ(result, stdResult);
}

void test_count_if()
{
    using Iter = bit_counting_iterator;
    using Sent = sentinel<std::int64_t>;

    auto predicate = [](std::int64_t v) { return v == 1; };
    auto stdResult = std::count_if(Iter{0}, Iter{33}, predicate);

    Iter::difference_type result = pika::ranges::count_if(
        pika::execution::seq, Iter{0}, Sent{33}, predicate);

    PIKA_TEST_EQ(result, stdResult);

    result = pika::ranges::count_if(
        pika::execution::par, Iter{0}, Sent{33}, predicate);

    PIKA_TEST_EQ(result, stdResult);
}

int pika_main()
{
    test_count();
    test_count_if();

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
