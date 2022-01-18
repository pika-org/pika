//  Copyright (c) 2014-2015 Anton Bikineev
//  Copyright (c) 2015 Andreas Schaefer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/modules/testing.hpp>
#include <pika/serialization/base_object.hpp>
#include <pika/serialization/serialize.hpp>
#include <pika/serialization/shared_ptr.hpp>

#include <iostream>
#include <memory>
#include <vector>

template <class T>
struct A
{
    int a;

    explicit A(int a)
      : a(a)
    {
    }

    A() = default;

    virtual ~A(){};

    virtual void foo() const = 0;
};

PIKA_TRAITS_NONINTRUSIVE_POLYMORPHIC_TEMPLATE(template <class T>, A<T>)

template <class T>
struct B : A<T>
{
    int b;

    explicit B(int b)
      : A<T>(b - 1)
      , b(b)
    {
    }

    B() = default;

    virtual ~B(){};

    void foo() const override {}

    PIKA_SERIALIZATION_POLYMORPHIC_TEMPLATE_SEMIINTRUSIVE(B)
};

PIKA_TRAITS_NONINTRUSIVE_POLYMORPHIC_TEMPLATE(template <class T>, B<T>)
PIKA_SERIALIZATION_REGISTER_CLASS_TEMPLATE(template <class T>, B<T>)

template <class S, class T>
struct C : A<T>
{
    int b;
    S c;

    explicit C(int b)
      : A<T>(b - 1)
      , b(b)
      , c(b + 1)
    {
    }
    C() = default;

    void foo() const override {}

    PIKA_SERIALIZATION_POLYMORPHIC_TEMPLATE_SEMIINTRUSIVE(C)
};

PIKA_TRAITS_NONINTRUSIVE_POLYMORPHIC_TEMPLATE(
    (template <class S, class T>), (C<S, T>) )
PIKA_SERIALIZATION_REGISTER_CLASS_TEMPLATE(
    (template <class S, class T>), (C<S, T>) )

namespace pika { namespace serialization {

    template <class Archive, class T>
    void serialize(Archive& archive, A<T>& s, unsigned)
    {
        archive& s.a;
    }

    template <class Archive, class T>
    void serialize(Archive& archive, B<T>& s, unsigned)
    {
        archive& pika::serialization::base_object<A<T>>(s);
        archive& s.b;
    }

    template <class Archive, class S, class T>
    void serialize(Archive& archive, C<S, T>& s, unsigned)
    {
        archive& pika::serialization::base_object<A<T>>(s);
        archive& s.b;
        archive& s.c;
    }

}}    // namespace pika::serialization

int main()
{
    std::vector<char> vector;
    {
        pika::serialization::output_archive archive{vector};
        std::shared_ptr<A<int>> b = std::make_shared<B<int>>(-4);
        std::shared_ptr<A<char>> c = std::make_shared<B<char>>(44);
        std::shared_ptr<A<double>> d = std::make_shared<B<double>>(99);
        std::shared_ptr<A<float>> e = std::make_shared<C<short, float>>(222);
        archive << b << c << d << e;
    }

    {
        pika::serialization::input_archive archive{vector};
        std::shared_ptr<A<int>> b;
        std::shared_ptr<A<char>> c;
        std::shared_ptr<A<double>> d;
        std::shared_ptr<A<float>> e;

        archive >> b;
        archive >> c;
        archive >> d;
        archive >> e;

        PIKA_TEST_EQ(b->a, -5);
        PIKA_TEST_EQ(std::static_pointer_cast<B<int>>(b)->b, -4);

        PIKA_TEST_EQ(c->a, 43);
        PIKA_TEST_EQ(std::static_pointer_cast<B<char>>(c)->b, 44);

        PIKA_TEST_EQ(d->a, 98);
        PIKA_TEST_EQ(std::static_pointer_cast<B<double>>(d)->b, 99);

        PIKA_TEST_EQ(e->a, 221);
        PIKA_TEST_EQ((std::static_pointer_cast<C<short, float>>(e)->b), 222);
        PIKA_TEST_EQ((std::static_pointer_cast<C<short, float>>(e)->c), 223);
    }

    return pika::util::report_errors();
}
