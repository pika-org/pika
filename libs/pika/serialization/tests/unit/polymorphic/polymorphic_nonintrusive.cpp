//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/serialization/base_object.hpp>
#include <pika/serialization/input_archive.hpp>
#include <pika/serialization/output_archive.hpp>
#include <pika/serialization/serialize.hpp>
#include <pika/serialization/shared_ptr.hpp>

#include <pika/modules/testing.hpp>

#include <memory>
#include <vector>

struct A
{
    explicit A(int a = 8)
      : a(a)
    {
    }
    virtual ~A() {}

    int a;
};

template <typename Archive>
void serialize(Archive& ar, A& a, unsigned)
{
    ar& a.a;
}

PIKA_SERIALIZATION_REGISTER_CLASS(A)
PIKA_TRAITS_NONINTRUSIVE_POLYMORPHIC(A)

struct B
{
    B()
      : b(6)
    {
    }
    explicit B(int i)
      : b(i)
    {
    }

    virtual ~B() {}

    virtual void f() = 0;

    int b;
};

template <class Archive>
void serialize(Archive& ar, B& b, unsigned)
{
    ar& b.b;
}

PIKA_TRAITS_NONINTRUSIVE_POLYMORPHIC(B)

struct D : B
{
    D()
      : d(89)
    {
    }
    explicit D(int i)
      : B(i)
      , d(89)
    {
    }
    void f() {}

    int d;
};

template <class Archive>
void serialize(Archive& ar, D& d, unsigned)
{
    d.b = 4711;
    ar& pika::serialization::base_object<B>(d);
    ar& d.d;
}

PIKA_SERIALIZATION_REGISTER_CLASS(D)

template <typename T>
struct C
{
    PIKA_SERIALIZATION_POLYMORPHIC_TEMPLATE_SEMIINTRUSIVE(C)

    explicit C(T c)
      : c(c)
    {
    }

    T c;
};

template <typename Archive, typename T>
void serialize(Archive& ar, C<T>& c, unsigned)
{
    ar& c.c;
}

template <typename T>
C<T>* c_factory(pika::serialization::input_archive& ar, C<T>* /*unused*/)
{
    C<T>* c = new C<T>(999);
    serialize(ar, *c, 0);
    return c;
}

PIKA_TRAITS_NONINTRUSIVE_POLYMORPHIC_TEMPLATE((template <class T>), C<T>)
PIKA_SERIALIZATION_REGISTER_CLASS_TEMPLATE(template <class T>, C<T>)
PIKA_SERIALIZATION_WITH_CUSTOM_CONSTRUCTOR_TEMPLATE(
    (template <typename T>), (C<T>), c_factory)

template <typename T>
struct E : public A
{
public:
    PIKA_SERIALIZATION_POLYMORPHIC_TEMPLATE_SEMIINTRUSIVE(E)

    E(int i, T t)
      : A(i)
      , c(t)
    {
    }

    C<T> c;
};

namespace pika { namespace serialization {

    template <class Archive, class T>
    void serialize(Archive& archive, E<T>& s, unsigned)
    {
        archive& pika::serialization::base_object<A>(s);
        archive& s.c;
    }
}}    // namespace pika::serialization

template <typename T>
E<T>* e_factory(pika::serialization::input_archive& ar, E<T>* /*unused*/)
{
    E<T>* e = new E<T>(99, 9999);
    serialize(ar, *e, 0);
    return e;
}

PIKA_TRAITS_NONINTRUSIVE_POLYMORPHIC_TEMPLATE((template <class T>), E<T>)
PIKA_SERIALIZATION_REGISTER_CLASS_TEMPLATE(template <class T>, E<T>)
PIKA_SERIALIZATION_WITH_CUSTOM_CONSTRUCTOR_TEMPLATE(
    (template <typename T>), (E<T>), e_factory)

void test_basic()
{
    std::vector<char> buffer;
    pika::serialization::output_archive oarchive(buffer);
    oarchive << A();
    D d;
    B const& b1 = d;
    oarchive << b1;

    pika::serialization::input_archive iarchive(buffer);
    A a;
    iarchive >> a;
    D d1;
    B& b2 = d1;
    iarchive >> b2;
    PIKA_TEST_EQ(a.a, 8);
    PIKA_TEST_EQ(&b2, &d1);
    PIKA_TEST_EQ(b2.b, d1.b);
    PIKA_TEST_EQ(d.b, d1.b);
    PIKA_TEST_EQ(d.d, d1.d);
}

void test_member()
{
    std::vector<char> buffer;
    {
        std::shared_ptr<A> struct_a(new E<float>(1, 2.3f));
        pika::serialization::output_archive oarchive(buffer);
        oarchive << struct_a;
    }
    {
        std::shared_ptr<A> struct_b;
        pika::serialization::input_archive iarchive(buffer);
        iarchive >> struct_b;
        PIKA_TEST_EQ(struct_b->a, 1);
        PIKA_TEST_EQ(dynamic_cast<E<float>*>(&*struct_b)->c.c, 2.3f);
    }
}

int main()
{
    test_basic();
    test_member();

    return pika::util::report_errors();
}
