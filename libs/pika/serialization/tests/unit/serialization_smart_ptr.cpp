//  Copyright (c) 2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// pikainspect:nodeprecatedinclude:boost/shared_ptr.hpp
// pikainspect:nodeprecatedname:boost::shared_ptr
// pikainspect:nodeprecatedinclude:boost/intrusive_ptr.hpp
// pikainspect:nodeprecatedname:boost::intrusive_ptr

#include <pika/serialization/input_archive.hpp>
#include <pika/serialization/intrusive_ptr.hpp>
#include <pika/serialization/output_archive.hpp>
#include <pika/serialization/serialize.hpp>
#include <pika/serialization/shared_ptr.hpp>
#include <pika/serialization/unique_ptr.hpp>

#include <pika/modules/testing.hpp>

#if defined(PIKA_SERIALIZATION_HAVE_BOOST_TYPES)
#include <boost/intrusive_ptr.hpp>
#include <boost/shared_ptr.hpp>
#endif

#include <memory>
#include <vector>

#if defined(PIKA_SERIALIZATION_HAVE_BOOST_TYPES)
void test_boost_shared()
{
    boost::shared_ptr<int> ip(new int(7));
    boost::shared_ptr<int> op1;
    boost::shared_ptr<int> op2;
    {
        std::vector<char> buffer;
        pika::serialization::output_archive oarchive(buffer);
        oarchive << ip << ip;

        pika::serialization::input_archive iarchive(buffer);
        iarchive >> op1;
        iarchive >> op2;
    }
    PIKA_TEST_NEQ(op1.get(), ip.get());
    PIKA_TEST_NEQ(op2.get(), ip.get());
    PIKA_TEST_EQ(op1.get(), op2.get());
    PIKA_TEST_EQ(*op1, *ip);
    op1.reset();
    PIKA_TEST_EQ(*op2, *ip);
}
#endif

void test_shared()
{
    std::shared_ptr<int> ip(new int(7));
    std::shared_ptr<int> op1;
    std::shared_ptr<int> op2;
    {
        std::vector<char> buffer;
        pika::serialization::output_archive oarchive(buffer);
        oarchive << ip << ip;

        pika::serialization::input_archive iarchive(buffer);
        iarchive >> op1;
        iarchive >> op2;
    }
    PIKA_TEST_NEQ(op1.get(), ip.get());
    PIKA_TEST_NEQ(op2.get(), ip.get());
    PIKA_TEST_EQ(op1.get(), op2.get());
    PIKA_TEST_EQ(*op1, *ip);
    op1.reset();
    PIKA_TEST_EQ(*op2, *ip);
}

void test_unique()
{
    std::unique_ptr<int> ip(new int(7));
    std::unique_ptr<int> op1;
    std::unique_ptr<int> op2;
    {
        std::vector<char> buffer;
        pika::serialization::output_archive oarchive(buffer);
        oarchive << ip << ip;

        pika::serialization::input_archive iarchive(buffer);
        iarchive >> op1;
        iarchive >> op2;
    }
    PIKA_TEST_NEQ(op1.get(), ip.get());
    PIKA_TEST_NEQ(op2.get(), ip.get());
    PIKA_TEST_NEQ(op1.get(), op2.get());    //untracked
    PIKA_TEST_EQ(*op1, *ip);
    PIKA_TEST_EQ(*op2, *ip);
}

#if defined(PIKA_SERIALIZATION_HAVE_BOOST_TYPES)
struct A
{
    A()
      : i(7)
      , count(0)
    {
    }
    int i;

    int count;

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        ar& i;
    }
};

void intrusive_ptr_add_ref(A* a)
{
    ++a->count;
}

void intrusive_ptr_release(A* a)
{
    if (--a->count == 0)
    {
        delete a;
    }
}

void test_intrusive()
{
    boost::intrusive_ptr<A> ip(new A());
    boost::intrusive_ptr<A> op1;
    boost::intrusive_ptr<A> op2;
    {
        std::vector<char> buffer;
        pika::serialization::output_archive oarchive(buffer);
        oarchive << ip << ip;

        pika::serialization::input_archive iarchive(buffer);
        iarchive >> op1;
        iarchive >> op2;
    }
    PIKA_TEST_EQ(ip->count, 1);
    PIKA_TEST_NEQ(op1.get(), ip.get());
    PIKA_TEST_NEQ(op2.get(), ip.get());
    PIKA_TEST_EQ(op1.get(), op2.get());
    PIKA_TEST_EQ(op1->i, ip->i);
    PIKA_TEST_EQ(op1->count, 2);
    PIKA_TEST_EQ(op2->count, 2);
    op1.reset();
    PIKA_TEST_EQ(op2->count, 1);
    PIKA_TEST_EQ(op2->i, ip->i);
}
#endif

int main()
{
    test_shared();
    test_unique();
#if defined(PIKA_SERIALIZATION_HAVE_BOOST_TYPES)
    test_boost_shared();
    test_intrusive();
#endif

    return pika::util::report_errors();
}
