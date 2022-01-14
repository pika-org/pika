//  Copyright (c) 2019 Piotr Mikolajczyk
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/algorithms/set_difference.hpp>
#include <pika/parallel/algorithms/set_intersection.hpp>

#include <string>
#include <vector>

void set_difference_small_test(int rounds)
{
    std::vector<int> set_a{1, 2, 3, 4, 5};
    std::vector<int> set_b{1, 2, 4};
    std::vector<int> a_minus_b(2);

    std::vector<int> perfect(2);
    std::set_difference(set_a.begin(), set_a.end(), set_b.begin(), set_b.end(),
        perfect.begin());

    while (--rounds)
    {
        pika::set_difference(pika::execution::par, set_a.begin(), set_a.end(),
            set_b.begin(), set_b.end(), a_minus_b.begin());
        PIKA_TEST(perfect == a_minus_b);
    }
}

void set_difference_medium_test(int rounds)
{
    std::vector<int> set_a(50);
    std::vector<int> set_b(20);

    std::iota(set_a.begin(), set_a.end(), 1);
    std::iota(set_b.begin(), set_b.end(), 2);

    std::vector<int> a_minus_b(50);

    std::vector<int> perfect(50);
    std::set_difference(set_a.begin(), set_a.end(), set_b.begin(), set_b.end(),
        perfect.begin());

    while (--rounds)
    {
        pika::set_difference(pika::execution::par, set_a.begin(), set_a.end(),
            set_b.begin(), set_b.end(), a_minus_b.begin());
        PIKA_TEST(perfect == a_minus_b);
    }
}

void set_difference_large_test(int rounds)
{
    std::vector<int> set_a(5000000);
    std::vector<int> set_b(3000000);

    std::iota(set_a.begin(), set_a.end(), 1);
    std::fill(set_b.begin(), set_b.begin() + 1000000, 1);
    std::iota(set_b.begin() + 1000000, set_b.end(), 2);

    std::vector<int> a_minus_b(5000000);

    std::vector<int> perfect(5000000);
    std::set_difference(set_a.begin(), set_a.end(), set_b.begin(), set_b.end(),
        perfect.begin());

    while (--rounds)
    {
        pika::set_difference(pika::execution::par, set_a.begin(), set_a.end(),
            set_b.begin(), set_b.end(), a_minus_b.begin());
        PIKA_TEST(perfect == a_minus_b);
    }
}

void set_difference_test(int rounds)
{
    set_difference_small_test(rounds);
    set_difference_medium_test(rounds);
    set_difference_large_test(rounds);
}

void set_intersection_small_test(int rounds)
{
    std::vector<int> set_a{1, 2, 3, 4, 5};
    std::vector<int> set_b{1, 2, 7};
    std::vector<int> a_and_b(2);

    std::vector<int> perfect(2);
    std::set_intersection(set_a.begin(), set_a.end(), set_b.begin(),
        set_b.end(), perfect.begin());

    while (--rounds)
    {
        pika::set_intersection(pika::execution::par, set_a.begin(), set_a.end(),
            set_b.begin(), set_b.end(), a_and_b.begin());
        PIKA_TEST(perfect == a_and_b);
    }
}

void set_intersection_medium_test(int rounds)
{
    std::vector<int> set_a(50);
    std::vector<int> set_b(20);

    std::iota(set_a.begin(), set_a.end(), 1);
    std::iota(set_b.begin(), set_b.end(), 2);

    std::vector<int> a_and_b(20);

    std::vector<int> perfect(20);
    std::set_intersection(set_a.begin(), set_a.end(), set_b.begin(),
        set_b.end(), perfect.begin());

    while (--rounds)
    {
        pika::set_intersection(pika::execution::par, set_a.begin(), set_a.end(),
            set_b.begin(), set_b.end(), a_and_b.begin());
        PIKA_TEST(perfect == a_and_b);
    }
}

void set_intersection_large_test(int rounds)
{
    std::vector<int> set_a(5000000);
    std::vector<int> set_b(3000000);

    std::iota(set_a.begin(), set_a.end(), 1);
    std::fill(set_b.begin(), set_b.begin() + 1000000, 1);
    std::iota(set_b.begin() + 1000000, set_b.end(), 2);

    std::vector<int> a_and_b(3000000);

    std::vector<int> perfect(3000000);
    std::set_intersection(set_a.begin(), set_a.end(), set_b.begin(),
        set_b.end(), perfect.begin());

    while (--rounds)
    {
        pika::set_intersection(pika::execution::par, set_a.begin(), set_a.end(),
            set_b.begin(), set_b.end(), a_and_b.begin());
        PIKA_TEST(perfect == a_and_b);
    }
}

void set_intersection_test(int rounds)
{
    set_intersection_small_test(rounds);
    set_intersection_medium_test(rounds);
    set_intersection_large_test(rounds);
}

int pika_main()
{
    int rounds = 5;
    set_intersection_test(rounds);
    set_difference_test(rounds);
    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    pika::local::init_params init_args;
    init_args.cfg = cfg;

    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv, init_args), 0,
        "pika main exted with non-zero status");

    return pika::util::report_errors();
}
