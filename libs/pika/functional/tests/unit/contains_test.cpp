//  Taken from the Boost.Function library

//  Copyright Douglas Gregor 2001-2003.
//  Copyright 2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <pika/functional/function.hpp>
#include <pika/modules/testing.hpp>

#include <functional>

static int forty_two()
{
    return 42;
}

struct Seventeen
{
    int operator()() const
    {
        return 17;
    }
};

//struct ReturnInt
//{
//    explicit ReturnInt(int value) : value(value) {}
//
//    int operator()() const { return value; }
//
//    int value;
//};
//
//bool operator==(const ReturnInt& x, const ReturnInt& y)
//{ return x.value == y.value; }
//
//bool operator!=(const ReturnInt& x, const ReturnInt& y)
//{ return x.value != y.value; }
//
//namespace contain_test {
//
//    struct ReturnIntFE
//    {
//        explicit ReturnIntFE(int value) : value(value) {}
//
//        int operator()() const { return value; }
//
//        int value;
//    };
//}

static void target_test()
{
    pika::util::function_nonser<int()> f;

    f = &forty_two;
    PIKA_TEST_EQ(*f.target<int (*)()>(), &forty_two);
    PIKA_TEST(!f.target<Seventeen>());

    f = Seventeen();
    PIKA_TEST(!f.target<int (*)()>());
    PIKA_TEST(f.target<Seventeen>());

    Seventeen this_seventeen;
    f = this_seventeen;
    PIKA_TEST(!f.target<int (*)()>());
    PIKA_TEST(f.target<Seventeen>());
}

//static void equal_test()
//{
//    pika::util::function_nonser<int()> f;
//
//    f = &forty_two;
//    PIKA_TEST(f == &forty_two);
//    PIKA_TEST(f != ReturnInt(17));
//    PIKA_TEST(&forty_two == f);
//    PIKA_TEST(ReturnInt(17) != f);
//
//    PIKA_TEST(f.contains(&forty_two));
//
//    f = ReturnInt(17);
//    PIKA_TEST(f != &forty_two);
//    PIKA_TEST(f == ReturnInt(17));
//    PIKA_TEST(f != ReturnInt(16));
//    PIKA_TEST(&forty_two != f);
//    PIKA_TEST(ReturnInt(17) == f);
//    PIKA_TEST(ReturnInt(16) != f);
//
//    PIKA_TEST(f.contains(ReturnInt(17)));
//
//    f = contain_test::ReturnIntFE(17);
//    PIKA_TEST(f != &forty_two);
//    PIKA_TEST(f == contain_test::ReturnIntFE(17));
//    PIKA_TEST(f != contain_test::ReturnIntFE(16));
//    PIKA_TEST(&forty_two != f);
//    PIKA_TEST(contain_test::ReturnIntFE(17) == f);
//    PIKA_TEST(contain_test::ReturnIntFE(16) != f);
//
//    PIKA_TEST(f.contains(contain_test::ReturnIntFE(17)));
//
//    pika::util::function_nonser<int(void)> g;
//
//    g = &forty_two;
//    PIKA_TEST(g == &forty_two);
//    PIKA_TEST(g != ReturnInt(17));
//    PIKA_TEST(&forty_two == g);
//    PIKA_TEST(ReturnInt(17) != g);
//
//    g = ReturnInt(17);
//    PIKA_TEST(g != &forty_two);
//    PIKA_TEST(g == ReturnInt(17));
//    PIKA_TEST(g != ReturnInt(16));
//    PIKA_TEST(&forty_two != g);
//    PIKA_TEST(ReturnInt(17) == g);
//    PIKA_TEST(ReturnInt(16) != g);
//}
//
//static void ref_equal_test()
//{
//    {
//        ReturnInt ri(17);
//        pika::util::function_nonser0<int> f = std::ref(ri);
//
//        // References and values are equal
//        PIKA_TEST(f == std::ref(ri));
//        PIKA_TEST(f == ri);
//        PIKA_TEST(std::ref(ri) == f);
//        PIKA_TEST(!(f != std::ref(ri)));
//        PIKA_TEST(!(f != ri));
//        PIKA_TEST(!(std::ref(ri) != f));
//        PIKA_TEST(ri == f);
//        PIKA_TEST(!(ri != f));
//
//        // Values equal, references inequal
//        ReturnInt ri2(17);
//        PIKA_TEST(f == ri2);
//        PIKA_TEST(f != std::ref(ri2));
//        PIKA_TEST(std::ref(ri2) != f);
//        PIKA_TEST(!(f != ri2));
//        PIKA_TEST(!(f == std::ref(ri2)));
//        PIKA_TEST(!(std::ref(ri2) == f));
//        PIKA_TEST(ri2 == f);
//        PIKA_TEST(!(ri2 != f));
//    }
//
//    {
//        ReturnInt ri(17);
//        pika::util::function_nonser<int(void)> f = std::ref(ri);
//
//        // References and values are equal
//        PIKA_TEST(f == std::ref(ri));
//        PIKA_TEST(f == ri);
//        PIKA_TEST(std::ref(ri) == f);
//        PIKA_TEST(!(f != std::ref(ri)));
//        PIKA_TEST(!(f != ri));
//        PIKA_TEST(!(std::ref(ri) != f));
//        PIKA_TEST(ri == f);
//        PIKA_TEST(!(ri != f));
//
//        // Values equal, references inequal
//        ReturnInt ri2(17);
//        PIKA_TEST(f == ri2);
//        PIKA_TEST(f != std::ref(ri2));
//        PIKA_TEST(std::ref(ri2) != f);
//        PIKA_TEST(!(f != ri2));
//        PIKA_TEST(!(f == std::ref(ri2)));
//        PIKA_TEST(!(std::ref(ri2) == f));
//        PIKA_TEST(ri2 == f);
//        PIKA_TEST(!(ri2 != f));
//    }
//}

int main(int, char*[])
{
    target_test();
    //    equal_test();
    //    ref_equal_test();

    return pika::util::report_errors();
}
