//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/string_util/from_string.hpp>
#include <pika/testing.hpp>
#include <pika/topology/cpu_mask.hpp>

#include <cstddef>
#include <set>
#include <string>
#include <vector>

struct test_input
{
    std::string str;
    std::size_t expected_size;
    std::set<std::size_t> expected_bits;
};

void test_valid()
{
    const std::vector<test_input> tests
    {
        {"0x0", 4, {}}, {"0x1", 4, {0}}, {"0x2", 4, {1}}, {"0xb", 4, {0, 1, 3}},
            {"0xf", 4, {0, 1, 2, 3}}, {"0xff", 2 * 4, {0, 1, 2, 3, 4, 5, 6, 7}},
            {"0x808080", 6 * 4, {7, 15, 23}},
            {"0x0f0f0f0f0f0f0f0f", 16 * 4,
                {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41,
                    42, 43, 48, 49, 50, 51, 56, 57, 58, 59}},
            {"0x0f0e0d0c04030201", 16 * 4,
                {0, 9, 16, 17, 26, 34, 35, 40, 42, 43, 49, 50, 51, 56, 57, 58, 59}},
            {"       0x808080     \t  ", 6 * 4, {7, 15, 23}},
#if !defined(PIKA_HAVE_MAX_CPU_COUNT)
            {"0xff00000000000000000000", 22 * 4, {80, 81, 82, 83, 84, 85, 86, 87}},
#endif
    };

    for (const auto& t : tests)
    {
        auto mask = pika::detail::from_string<pika::threads::detail::mask_type>(t.str);
#if defined(PIKA_HAVE_MAX_CPU_COUNT)
# if defined(PIKA_HAVE_MORE_THAN_64_THREADS)
        PIKA_TEST_EQ(pika::threads::detail::mask_size(mask), std::size_t(PIKA_HAVE_MAX_CPU_COUNT));
# else
        PIKA_TEST_EQ(pika::threads::detail::mask_size(mask), std::size_t(64));
# endif
#else
        PIKA_TEST_EQ(pika::threads::detail::mask_size(mask), t.expected_size);
#endif
        for (std::size_t i = 0; i < pika::threads::detail::mask_size(mask); ++i)
        {
            if (pika::threads::detail::test(mask, i))
            {
                PIKA_TEST(t.expected_bits.find(i) != t.expected_bits.cend());
            }
            else { PIKA_TEST(t.expected_bits.find(i) == t.expected_bits.cend()); }
        }
    }
}

void test_invalid()
{
    const std::vector<std::string> tests{
        "", "0", "0x", "Ob", "foobar", "x0xff", "0xffg", " 0xabcdefgh\t"};

    for (const auto& t : tests)
    {
        try
        {
            (void) pika::detail::from_string<pika::threads::detail::mask_type>(t);
            PIKA_TEST(false);
        }
        catch (pika::detail::bad_lexical_cast const&)
        {
            PIKA_TEST(true);
        }
        catch (...)
        {
            PIKA_TEST(false);
        }
    }
}

int main()
{
    test_valid();
    test_invalid();

    return 0;
}
