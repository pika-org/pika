// Copyright (C) 1999, 2000 Jaakko Jarvi (jaakko.jarvi@cs.utu.fi)
// Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

// For more information, see http://www.boost.org

//  tuple_test_bench.cpp  --------------------------------

// clang-format off
#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wdouble-promotion"
#elif defined (__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wdouble-promotion"
#endif
// clang-format on

#include <pika/local/config.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/modules/testing.hpp>

// clang-format off
#if defined(__clang__)
#  pragma clang diagnostic pop
#elif defined (__GNUC__)
#  pragma GCC diagnostic pop
#endif
// clang-format on

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

// ----------------------------------------------------------------------------
// helpers
// ----------------------------------------------------------------------------

class A
{
};
class B
{
};
class C
{
};

// classes with different kinds of conversions
class AA
{
};
class BB : public AA
{
};
struct CC
{
    CC() {}
    CC(const BB&) {}
};
struct DD
{
    operator CC() const
    {
        return CC();
    };
};

// something to prevent warnings for unused variables
template <class T>
void dummy(const T&)
{
}

// no public default constructor
class foo
{
public:
    explicit foo(int v)
      : val(v)
    {
    }

    bool operator==(const foo& other) const
    {
        return val == other.val;
    }

private:
    foo() {}
    int val;
};

// another class without a public default constructor
class no_def_constructor
{
    no_def_constructor() {}

public:
    no_def_constructor(std::string) {}
};

// A non-copyable class
class no_copy
{
    no_copy(const no_copy&) {}

public:
    no_copy(){};
};

// ----------------------------------------------------------------------------
// Testing different element types --------------------------------------------
// ----------------------------------------------------------------------------

typedef pika::tuple<int> t1;
typedef pika::tuple<double&, const double&, const double, double*, const double*>
    t2;
typedef pika::tuple<A, int (*)(char, int), C> t3;
typedef pika::tuple<std::string, std::pair<A, B>> t4;
typedef pika::tuple<A*, pika::tuple<const A*, const B&, C>, bool, void*> t5;
typedef pika::tuple<volatile int, const volatile char&, int (&)(float)> t6;
typedef pika::tuple<B (A::*)(C&), A&> t7;

// -----------------------------------------------------------------------
// -tuple construction tests ---------------------------------------------
// -----------------------------------------------------------------------

no_copy y;
pika::tuple<no_copy&> x = pika::tuple<no_copy&>(y);    // ok

char cs[10];
pika::tuple<char (&)[10]> v2(cs);    // ok

void construction_test()
{
    pika::tuple<int> t1;
    PIKA_TEST_EQ(pika::get<0>(t1), int());

    pika::tuple<float> t2(5.5f);
    PIKA_TEST_RANGE(pika::get<0>(t2), 5.4f, 5.6f);

    pika::tuple<foo> t3(foo(12));
    PIKA_TEST(pika::get<0>(t3) == foo(12));

    pika::tuple<double> t4(t2);
    PIKA_TEST_RANGE(pika::get<0>(t4), 5.4f, 5.6f);

    pika::tuple<int, float> t5;
    PIKA_TEST_EQ(pika::get<0>(t5), int());
    PIKA_TEST_EQ(pika::get<1>(t5), float());

    pika::tuple<int, float> t6(12, 5.5f);
    PIKA_TEST_EQ(pika::get<0>(t6), 12);
    PIKA_TEST_RANGE(pika::get<1>(t6), 5.4f, 5.6f);

    pika::tuple<int, float> t7(t6);
    PIKA_TEST_EQ(pika::get<0>(t7), 12);
    PIKA_TEST_RANGE(pika::get<1>(t7), 5.4f, 5.6f);

    pika::tuple<long, double> t8(t6);
    PIKA_TEST_EQ(pika::get<0>(t8), 12);
    PIKA_TEST_RANGE(pika::get<1>(t8), 5.4f, 5.6f);

    dummy(pika::tuple<no_def_constructor, no_def_constructor,
        no_def_constructor>(std::string("Jaba"),    // ok, since the default
        std::string("Daba"),                        // constructor is not used
        std::string("Doo")));

    // testing default values
    dummy(pika::tuple<int, double>());
    dummy(pika::tuple<int, double>(1, 3.14));

    //dummy(pika::tuple<double&>()); // should fail, not defaults for references
    //dummy(pika::tuple<const double&>()); // likewise

    double dd = 5;
    dummy(pika::tuple<double&>(dd));    // ok

    dummy(pika::tuple<const double&>(dd + 3.14));    // ok, but dangerous

    //dummy(pika::tuple<double&>(dd+3.14)); // should fail,
    // temporary to non-const reference
}

// ----------------------------------------------------------------------------
// - testing element access ---------------------------------------------------
// ----------------------------------------------------------------------------

void element_access_test()
{
    double d = 2.7;
    A a;
    pika::tuple<int, double&, const A&, int> t(1, d, a, 2);
    const pika::tuple<int, double&, const A, int> ct = t;

    int i = pika::get<0>(t);
    int i2 = pika::get<3>(t);

    PIKA_TEST(i == 1 && i2 == 2);

    int j = pika::get<0>(ct);
    PIKA_TEST_EQ(j, 1);

    PIKA_TEST(pika::get<0>(t) = 5);

    //pika::get<0>(ct) = 5; // can't assign to const

    double e = pika::get<1>(t);
    PIKA_TEST_RANGE(e, 2.69, 2.71);

    pika::get<1>(t) = 3.14 + i;
    PIKA_TEST_RANGE(pika::get<1>(t), 4.13, 4.15);

    //pika::get<2>(t) = A(); // can't assign to const
    //dummy(pika::get<4>(ct)); // illegal index

    ++pika::get<0>(t);
    PIKA_TEST_EQ(pika::get<0>(t), 6);

    PIKA_TEST((std::is_const<
                  pika::tuple_element<0, pika::tuple<int, float>>::type>::value !=
        true));
    PIKA_TEST((std::is_const<
        pika::tuple_element<0, const pika::tuple<int, float>>::type>::value));

    PIKA_TEST((std::is_const<
                  pika::tuple_element<1, pika::tuple<int, float>>::type>::value !=
        true));
    PIKA_TEST((std::is_const<
        pika::tuple_element<1, const pika::tuple<int, float>>::type>::value));

    PIKA_TEST((std::is_same<pika::tuple_element<1, std::array<float, 4>>::type,
        float>::value));

    dummy(i);
    dummy(i2);
    dummy(j);
    dummy(e);    // avoid warns for unused variables
}

// ----------------------------------------------------------------------------
// - copying tuples -----------------------------------------------------------
// ----------------------------------------------------------------------------

void copy_test()
{
    pika::tuple<int, char> t1(4, 'a');
    pika::tuple<int, char> t2(5, 'b');
    t2 = t1;
    PIKA_TEST_EQ(pika::get<0>(t1), pika::get<0>(t2));
    PIKA_TEST_EQ(pika::get<1>(t1), pika::get<1>(t2));

    pika::tuple<long, std::string> t3(2, "a");
    t3 = t1;
    PIKA_TEST_EQ((double) pika::get<0>(t1), pika::get<0>(t3));
    PIKA_TEST_EQ(pika::get<1>(t1), pika::get<1>(t3)[0]);

    // testing copy and assignment with implicit conversions between elements
    // testing tie

    pika::tuple<char, BB*, BB, DD> t;
    pika::tuple<int, AA*, CC, CC> a(t);
    a = t;

    int i;
    char c;
    double d;
    pika::tie(i, c, d) = pika::make_tuple(1, 'a', 5.5);

    PIKA_TEST_EQ(i, 1);
    PIKA_TEST_EQ(c, 'a');
    PIKA_TEST_RANGE(d, 5.4, 5.6);
}

void mutate_test()
{
    pika::tuple<int, float, bool, foo> t1(5, 12.2f, true, foo(4));
    pika::get<0>(t1) = 6;
    pika::get<1>(t1) = 2.2f;
    pika::get<2>(t1) = false;
    pika::get<3>(t1) = foo(5);

    PIKA_TEST_EQ(pika::get<0>(t1), 6);
    PIKA_TEST_RANGE(pika::get<1>(t1), 2.1f, 2.3f);
    PIKA_TEST_EQ(pika::get<2>(t1), false);
    PIKA_TEST(pika::get<3>(t1) == foo(5));
}

// ----------------------------------------------------------------------------
// make_tuple tests -----------------------------------------------------------
// ----------------------------------------------------------------------------

void make_tuple_test()
{
    pika::tuple<int, char> t1 = pika::make_tuple(5, 'a');
    PIKA_TEST_EQ(pika::get<0>(t1), 5);
    PIKA_TEST_EQ(pika::get<1>(t1), 'a');

    pika::tuple<int, std::string> t2;
    t2 = pika::make_tuple((short int) 2, std::string("Hi"));
    PIKA_TEST_EQ(pika::get<0>(t2), 2);
    PIKA_TEST_EQ(pika::get<1>(t2), "Hi");

    A a = A();
    B b;
    const A ca = a;
    pika::make_tuple(std::cref(a), b);
    pika::make_tuple(std::ref(a), b);
    pika::make_tuple(std::ref(a), std::cref(b));

    pika::make_tuple(std::ref(ca));

    // the result of make_tuple is assignable:
    PIKA_TEST(pika::make_tuple(2, 4, 6) ==
        (pika::make_tuple(1, 2, 3) = pika::make_tuple(2, 4, 6)));

    pika::make_tuple("Donald", "Daisy");    // should work;

    // You can store a reference to a function in a tuple
    pika::tuple<void (&)()> adf(make_tuple_test);

    dummy(adf);    // avoid warning for unused variable

    // But make_tuple doesn't work (in C++03)
    // with function references, since it creates a const qualified function type

    pika::make_tuple(make_tuple_test);

    // With function pointers, make_tuple works just fine

    pika::make_tuple(&make_tuple_test);

    // wrapping it the function reference with ref

    // pika::make_tuple(ref(foo3));
}

void tie_test()
{
    int a;
    char b;
    foo c(5);

    pika::tie(a, b, c) = pika::make_tuple(2, 'a', foo(3));
    PIKA_TEST_EQ(a, 2);
    PIKA_TEST_EQ(b, 'a');
    PIKA_TEST(c == foo(3));

    pika::tie(a, pika::ignore, c) = pika::make_tuple((short int) 5, false, foo(5));
    PIKA_TEST_EQ(a, 5);
    PIKA_TEST_EQ(b, 'a');
    PIKA_TEST(c == foo(5));

    // testing assignment from std::pair
    int i, j;
    pika::tie(i, j) = std::make_pair(1, 2);
    PIKA_TEST(i == 1 && j == 2);

    pika::tuple<int, int, float> ta;
    //ta = std::make_pair(1, 2); // should fail, tuple is of length 3, not 2

    dummy(ta);
}

// ----------------------------------------------------------------------------
// - testing cat -----------------------------------------------------------
// ----------------------------------------------------------------------------
void tuple_cat_test()
{
    pika::tuple<int, float> two = pika::make_tuple(1, 2.f);

    // Cat two tuples
    {
        pika::tuple<int, float, int, float> res = pika::tuple_cat(two, two);

        auto expected = pika::make_tuple(1, 2.f, 1, 2.f);

        PIKA_TEST(res == expected);
    }

    // Cat multiple tuples
    {
        pika::tuple<int, float, int, float, int, float> res =
            pika::tuple_cat(two, two, two);

        auto expected = pika::make_tuple(1, 2.f, 1, 2.f, 1, 2.f);

        PIKA_TEST(res == expected);
    }

    // Cat move only types
    {
        auto t0 = pika::make_tuple(std::unique_ptr<int>(new int(0)));
        auto t1 = pika::make_tuple(std::unique_ptr<int>(new int(1)));
        auto t2 = pika::make_tuple(std::unique_ptr<int>(new int(2)));

        pika::tuple<std::unique_ptr<int>, std::unique_ptr<int>,
            std::unique_ptr<int>>
            result =
                pika::tuple_cat(std::move(t0), std::move(t1), std::move(t2));

        PIKA_TEST_EQ((*pika::get<0>(result)), 0);
        PIKA_TEST_EQ((*pika::get<1>(result)), 1);
        PIKA_TEST_EQ((*pika::get<2>(result)), 2);
    }

    // Don't move references unconditionally (copyable types)
    {
        int i1 = 11;
        int i2 = 22;

        pika::tuple<int&> f1 = pika::forward_as_tuple(i1);
        pika::tuple<int&&> f2 = pika::forward_as_tuple(std::move(i2));

        pika::tuple<int&, int&&> result =
            pika::tuple_cat(std::move(f1), std::move(f2));

        PIKA_TEST_EQ((pika::get<0>(result)), 11);
        PIKA_TEST_EQ((pika::get<1>(result)), 22);
    }

    // Don't move references unconditionally (move only types)
    {
        std::unique_ptr<int> i1(new int(11));
        std::unique_ptr<int> i2(new int(22));

        pika::tuple<std::unique_ptr<int>&> f1 = pika::forward_as_tuple(i1);
        pika::tuple<std::unique_ptr<int>&&> f2 =
            pika::forward_as_tuple(std::move(i2));

        pika::tuple<std::unique_ptr<int>&, std::unique_ptr<int>&&> result =
            pika::tuple_cat(std::move(f1), std::move(f2));

        PIKA_TEST_EQ((*pika::get<0>(result)), 11);
        PIKA_TEST_EQ((*pika::get<1>(result)), 22);
    }
}

// ----------------------------------------------------------------------------
// - testing tuple equality   -------------------------------------------------
// ----------------------------------------------------------------------------

void equality_test()
{
    pika::tuple<int, char> t1(5, 'a');
    pika::tuple<int, char> t2(5, 'a');
    PIKA_TEST(t1 == t2);

    pika::tuple<int, char> t3(5, 'b');
    pika::tuple<int, char> t4(2, 'a');
    PIKA_TEST(t1 != t3);
    PIKA_TEST(t1 != t4);
    PIKA_TEST(!(t1 != t2));
}

// ----------------------------------------------------------------------------
// - testing tuple comparisons  -----------------------------------------------
// ----------------------------------------------------------------------------

void ordering_test()
{
    pika::tuple<int, float> t1(4, 3.3f);
    pika::tuple<short, float> t2(5, 3.3f);
    pika::tuple<long, double> t3(5, 4.4);
    PIKA_TEST(t1 < t2);
    PIKA_TEST(t1 <= t2);
    PIKA_TEST(t2 > t1);
    PIKA_TEST(t2 >= t1);
    PIKA_TEST(t2 < t3);
    PIKA_TEST(t2 <= t3);
    PIKA_TEST(t3 > t2);
    PIKA_TEST(t3 >= t2);
}

// ----------------------------------------------------------------------------
// - testing const tuples -----------------------------------------------------
// ----------------------------------------------------------------------------
void const_tuple_test()
{
    const pika::tuple<int, float> t1(5, 3.3f);
    PIKA_TEST_EQ(pika::get<0>(t1), 5);
    PIKA_TEST_EQ(pika::get<1>(t1), 3.3f);
}

// ----------------------------------------------------------------------------
// - testing length -----------------------------------------------------------
// ----------------------------------------------------------------------------
void tuple_length_test()
{
    typedef pika::tuple<int, float, double> t1;
    typedef pika::tuple<> t2;

    PIKA_TEST_EQ(pika::tuple_size<t1>::value, std::size_t(3));
    PIKA_TEST_EQ(pika::tuple_size<t2>::value, std::size_t(0));

    {
        using t3 = std::array<int, 4>;
        PIKA_TEST_EQ(pika::tuple_size<t3>::value, std::size_t(4));
    }
}

// ----------------------------------------------------------------------------
// - testing swap -----------------------------------------------------------
// ----------------------------------------------------------------------------
void tuple_swap_test()
{
    using std::swap;

    pika::tuple<int, float, double> t1(1, 2.0f, 3.0), t2(4, 5.0f, 6.0);
    swap(t1, t2);
    PIKA_TEST_EQ(pika::get<0>(t1), 4);
    PIKA_TEST_EQ(pika::get<1>(t1), 5.0f);
    PIKA_TEST_EQ(pika::get<2>(t1), 6.0);
    PIKA_TEST_EQ(pika::get<0>(t2), 1);
    PIKA_TEST_EQ(pika::get<1>(t2), 2.0f);
    PIKA_TEST_EQ(pika::get<2>(t2), 3.0);

    int i = 1, j = 2;

    pika::tuple<int&> t3(i), t4(j);
    swap(t3, t4);
    PIKA_TEST_EQ(pika::get<0>(t3), 2);
    PIKA_TEST_EQ(pika::get<0>(t4), 1);
    PIKA_TEST_EQ(i, 2);
    PIKA_TEST_EQ(j, 1);
}

void tuple_std_test()
{
#if defined(PIKA_DATASTRUCTURES_HAVE_ADAPT_STD_TUPLE)
    pika::tuple<int, float, double> t1(1, 2.0f, 3.0);
    std::tuple<int, float, double> t2 = t1;
    pika::tuple<int, float, double> t3 = t2;
    PIKA_TEST_EQ(std::get<0>(t1), 1);
    PIKA_TEST_EQ(std::get<0>(t2), 1);
    PIKA_TEST_EQ(std::get<0>(t3), 1);

    PIKA_TEST_EQ(pika::get<0>(t1), 1);
    PIKA_TEST_EQ(pika::get<0>(t2), 1);
    PIKA_TEST_EQ(pika::get<0>(t3), 1);

    PIKA_TEST_EQ(std::get<1>(t1), 2.0f);
    PIKA_TEST_EQ(std::get<1>(t2), 2.0f);
    PIKA_TEST_EQ(std::get<1>(t3), 2.0f);

    PIKA_TEST_EQ(pika::get<1>(t1), 2.0f);
    PIKA_TEST_EQ(pika::get<1>(t2), 2.0f);
    PIKA_TEST_EQ(pika::get<1>(t3), 2.0f);

    PIKA_TEST_EQ(std::get<2>(t1), 3.0);
    PIKA_TEST_EQ(std::get<2>(t2), 3.0);
    PIKA_TEST_EQ(std::get<2>(t3), 3.0);

    PIKA_TEST_EQ(pika::get<2>(t1), 3.0);
    PIKA_TEST_EQ(pika::get<2>(t2), 3.0);
    PIKA_TEST_EQ(pika::get<2>(t3), 3.0);
#endif
}

void tuple_structured_binding_test()
{
#ifdef PIKA_DATASTRUCTURES_HAVE_ADAPT_STD_TUPLE
    auto [a1, a2] = pika::make_tuple(1, '2');

    PIKA_TEST_EQ(a1, 1);
    PIKA_TEST_EQ(a2, '2');
#endif
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    {
        construction_test();
        element_access_test();
        copy_test();
        mutate_test();
        make_tuple_test();
        tie_test();
        tuple_cat_test();
        equality_test();
        ordering_test();
        const_tuple_test();
        tuple_length_test();
        tuple_swap_test();
        tuple_structured_binding_test();
    }

    return pika::util::report_errors();
}
