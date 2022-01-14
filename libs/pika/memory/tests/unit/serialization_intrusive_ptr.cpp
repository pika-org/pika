//  Copyright (c) 2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/memory/serialization/intrusive_ptr.hpp>
#include <pika/modules/memory.hpp>
#include <pika/modules/serialization.hpp>
#include <pika/modules/testing.hpp>

#include <memory>
#include <vector>

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
    pika::intrusive_ptr<A> ip(new A());
    pika::intrusive_ptr<A> op1;
    pika::intrusive_ptr<A> op2;

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

int main()
{
    test_intrusive();

    return pika::util::report_errors();
}
