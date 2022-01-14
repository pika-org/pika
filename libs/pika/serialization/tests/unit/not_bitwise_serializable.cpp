//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// enforce that types are bitwise serializable by default
#define PIKA_SERIALIZATION_HAVE_ALL_TYPES_ARE_BITWISE_SERIALIZABLE

#include <pika/local/config.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/serialization.hpp>
#include <pika/modules/testing.hpp>

#include <cstddef>
#include <vector>

bool serialize_A = false;
bool serialize_B = false;

struct A
{
    int a;
    double b;

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        // clang-format off
        ar & a & b;
        // clang-format on

        serialize_A = true;
    }
};

struct A_nonser
{
    int a;
    double b;
};

struct A_non_default_constructible
{
    A_non_default_constructible() = delete;
    A_non_default_constructible(int a, double b)
      : a(a)
      , b(b)
    {
    }

    int get_a() const
    {
        return a;
    }
    double get_b() const
    {
        return b;
    }

private:
    int a;
    double b;
};

template <typename Archive>
void save_construct_data(
    Archive& ar, A_non_default_constructible const* t, unsigned)
{
    ar << t->get_a() << t->get_b();
}

template <typename Archive>
void load_construct_data(Archive& ar, A_non_default_constructible* t, unsigned)
{
    int a;
    double b;
    ar >> a >> b;
    ::new (t) A_non_default_constructible(a, b);
}

struct B
{
    int a;
    double b;

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        // clang-format off
        ar & a & b;
        // clang-format on

        serialize_B = true;
    }
};

PIKA_IS_NOT_BITWISE_SERIALIZABLE(B)

int pika_main()
{
    {
        std::vector<char> buffer;
        pika::serialization::output_archive oarchive(buffer);

        serialize_A = false;
        A oa{42, 42.0};
        oarchive << oa;
        PIKA_TEST(serialize_A);

        pika::serialization::input_archive iarchive(buffer);
        A ia{0, 0.0};

        serialize_A = false;
        iarchive >> ia;
        PIKA_TEST(serialize_A);

        PIKA_TEST_EQ(ia.a, 42);
        PIKA_TEST_EQ(ia.b, 42.0);
    }

    {
        std::vector<char> buffer;
        pika::serialization::output_archive oarchive(buffer);

        std::vector<A_non_default_constructible> oa;
        oa.push_back(A_non_default_constructible(42, 42.0));
        oarchive << oa;

        pika::serialization::input_archive iarchive(buffer);
        std::vector<A_non_default_constructible> ia;

        iarchive >> ia;

        PIKA_TEST_EQ(ia.size(), std::size_t(1));
        PIKA_TEST_EQ(ia[0].get_a(), 42);
        PIKA_TEST_EQ(ia[0].get_b(), 42.0);
    }

    {
        std::vector<char> buffer;
        pika::serialization::output_archive oarchive(buffer);

        A_nonser oa{42, 42.0};
        oarchive << oa;

        pika::serialization::input_archive iarchive(buffer);
        A_nonser ia{0, 0.0};

        iarchive >> ia;

        PIKA_TEST_EQ(ia.a, 42);
        PIKA_TEST_EQ(ia.b, 42.0);
    }

    {
        std::vector<char> buffer;
        pika::serialization::output_archive oarchive(buffer);

        serialize_B = false;
        B ob{42, 42.0};
        oarchive << ob;
        PIKA_TEST(serialize_B);

        pika::serialization::input_archive iarchive(buffer);
        B ib{0, 0.0};

        serialize_B = false;
        iarchive >> ib;
        PIKA_TEST(serialize_B);

        PIKA_TEST_EQ(ib.a, 42);
        PIKA_TEST_EQ(ib.b, 42.0);
    }

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    pika::local::init(pika_main, argc, argv);
    return pika::util::report_errors();
}
