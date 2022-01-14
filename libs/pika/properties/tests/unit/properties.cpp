//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/modules/properties.hpp>
#include <pika/modules/testing.hpp>

struct property1
{
    int v = 0;
};

inline constexpr struct with_property_t : pika::functional::tag<with_property_t>
{
} with_property;

struct type1
{
    property1 p1{};
};

struct type2
{
    property1 p1{};
};

struct type3
{
    property1 p1{};
};

struct type4
{
    property1 p1{};
};

type1 tag_invoke(with_property_t, type1 const& t, property1 p)
{
    auto tt = t;
    tt.p1 = p;
    return tt;
}

type3 tag_invoke(
    pika::experimental::prefer_t, with_property_t, type3 const& t, property1 p)
{
    auto tt = t;
    tt.p1 = p;
    ++tt.p1.v;
    return tt;
}

type4 tag_invoke(
    pika::experimental::prefer_t, with_property_t, type4 const& t, property1 p)
{
    auto tt = t;
    tt.p1 = p;
    ++tt.p1.v;
    return tt;
}

type4 tag_invoke(with_property_t, type4 const& t, property1 p)
{
    auto tt = t;
    tt.p1 = p;
    return tt;
}

int main()
{
    // This property can be required, and thus also preferred
    static_assert(pika::is_invocable<with_property_t, type1, property1>::value,
        "Should be invocable");

    type1 t1_1{};

    type1 t1_2 = with_property(t1_1, property1{1});
    PIKA_TEST_EQ(t1_1.p1.v, 0);
    PIKA_TEST_EQ(t1_2.p1.v, 1);

    property1 p1_1{2};
    type1 t1_3 = with_property(t1_2, p1_1);
    PIKA_TEST_EQ(t1_2.p1.v, 1);
    PIKA_TEST_EQ(t1_3.p1.v, 2);

    property1 const p1_2{3};
    type1 t1_4 = with_property(t1_2, p1_2);
    PIKA_TEST_EQ(t1_3.p1.v, 2);
    PIKA_TEST_EQ(t1_4.p1.v, 3);

    static_assert(pika::is_invocable<pika::experimental::prefer_t,
                      with_property_t, type1, property1>::value,
        "Should be invocable");

    type1 t1_5 = pika::experimental::prefer(with_property, t1_4, property1{4});
    PIKA_TEST_EQ(t1_4.p1.v, 3);
    PIKA_TEST_EQ(t1_5.p1.v, 4);

    property1 p1_3{5};
    type1 t1_6 = pika::experimental::prefer(with_property, t1_5, p1_3);
    PIKA_TEST_EQ(t1_5.p1.v, 4);
    PIKA_TEST_EQ(t1_6.p1.v, 5);

    property1 const p1_4{6};
    type1 t1_7 = pika::experimental::prefer(with_property, t1_6, p1_4);
    PIKA_TEST_EQ(t1_6.p1.v, 5);
    PIKA_TEST_EQ(t1_7.p1.v, 6);

    // This property cannot be required, but can be preferred
    static_assert(!pika::is_invocable<with_property_t, type2, property1>::value,
        "Should not be invocable");

    static_assert(pika::is_invocable<pika::experimental::prefer_t,
                      with_property_t, type2, property1>::value,
        "Should be invocable");

    type2 t2_1{};

    type2 t2_2 = pika::experimental::prefer(with_property, t2_1, property1{7});
    PIKA_TEST_EQ(t2_1.p1.v, 0);
    PIKA_TEST_EQ(t2_2.p1.v, 0);

    property1 p1_5{8};
    type2 t2_3 = pika::experimental::prefer(with_property, t2_2, p1_5);
    PIKA_TEST_EQ(t2_2.p1.v, 0);
    PIKA_TEST_EQ(t2_3.p1.v, 0);

    property1 const p1_6{9};
    type2 t2_4 = pika::experimental::prefer(with_property, t2_3, p1_6);
    PIKA_TEST_EQ(t2_3.p1.v, 0);
    PIKA_TEST_EQ(t2_4.p1.v, 0);

    // This property cannot be required, but can be preferred. The prefer
    // functionality has been customized (it adds one to the passed property).
    static_assert(!pika::is_invocable<with_property_t, type3, property1>::value,
        "Should not be invocable");

    static_assert(pika::is_invocable<pika::experimental::prefer_t,
                      with_property_t, type3, property1>::value,
        "Should be invocable");

    type3 t3_1{};

    type3 t3_2 = pika::experimental::prefer(with_property, t3_1, property1{7});
    PIKA_TEST_EQ(t3_1.p1.v, 0);
    PIKA_TEST_EQ(t3_2.p1.v, 8);

    property1 p1_7{8};
    type3 t3_3 = pika::experimental::prefer(with_property, t3_2, p1_7);
    PIKA_TEST_EQ(t3_2.p1.v, 8);
    PIKA_TEST_EQ(t3_3.p1.v, 9);

    property1 const p1_8{9};
    type3 t3_4 = pika::experimental::prefer(with_property, t3_3, p1_8);
    PIKA_TEST_EQ(t3_3.p1.v, 9);
    PIKA_TEST_EQ(t3_4.p1.v, 10);

    // This property can be required and preferred through a customization. The
    // customization for prefer should take precedence over the require
    // customization.
    static_assert(pika::is_invocable<with_property_t, type4, property1>::value,
        "Should be invocable");

    static_assert(pika::is_invocable<pika::experimental::prefer_t,
                      with_property_t, type4, property1>::value,
        "Should be invocable");

    type4 t4_1{};

    type4 t4_2 = pika::experimental::prefer(with_property, t4_1, property1{10});
    PIKA_TEST_EQ(t4_1.p1.v, 0);
    PIKA_TEST_EQ(t4_2.p1.v, 11);

    property1 p1_9{11};
    type4 t4_3 = pika::experimental::prefer(with_property, t4_2, p1_9);
    PIKA_TEST_EQ(t4_2.p1.v, 11);
    PIKA_TEST_EQ(t4_3.p1.v, 12);

    property1 const p1_10{12};
    type4 t4_4 = pika::experimental::prefer(with_property, t4_3, p1_10);
    PIKA_TEST_EQ(t4_3.p1.v, 12);
    PIKA_TEST_EQ(t4_4.p1.v, 13);

    return pika::util::report_errors();
}
