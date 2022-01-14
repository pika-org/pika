//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// enforce pointers being serializable
#define PIKA_SERIALIZATION_HAVE_ALLOW_RAW_POINTER_SERIALIZATION

// allow for const tuple members
#define PIKA_SERIALIZATION_HAVE_ALLOW_CONST_TUPLE_MEMBERS

#include <pika/local/config.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/local/init.hpp>
#include <pika/serialization/input_archive.hpp>
#include <pika/serialization/output_archive.hpp>
#include <pika/serialization/serialize.hpp>
#include <pika/serialization/std_tuple.hpp>
#include <pika/serialization/tuple.hpp>

#include <pika/modules/testing.hpp>

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

// non-bitwise copyable type
struct A
{
    A() = default;

    template <typename Archive>
    void serialize(Archive&, unsigned)
    {
    }

    friend bool operator==(A const&, A const&)
    {
        return true;
    }
};

int pika_main()
{
    // serialize raw pointer
    {
        static_assert(pika::traits::is_bitwise_serializable_v<int*>);

        int value = 42;
        int* outp = &value;

        std::vector<char> buffer;
        std::vector<pika::serialization::serialization_chunk> chunks;
        pika::serialization::output_archive oarchive(buffer, 0, &chunks);
        oarchive << outp;
        std::size_t size = oarchive.bytes_written();

        pika::serialization::input_archive iarchive(buffer, size, &chunks);
        int* inp = nullptr;
        iarchive >> inp;
        PIKA_TEST(outp == inp);
    }

    // serialize raw pointer
    {
        static_assert(pika::traits::is_bitwise_serializable_v<int const*>);

        int value = 42;
        int const* outp = &value;

        std::vector<char> buffer;
        std::vector<pika::serialization::serialization_chunk> chunks;
        pika::serialization::output_archive oarchive(buffer, 0, &chunks);
        oarchive << outp;
        std::size_t size = oarchive.bytes_written();

        pika::serialization::input_archive iarchive(buffer, size, &chunks);
        int const* inp = nullptr;
        iarchive >> inp;
        PIKA_TEST(outp == inp);
    }

    // serialize raw pointer as part of tuple
    {
        static_assert(pika::traits::is_bitwise_serializable_v<
            pika::tuple<int*, int const>>);

        int value = 42;
        pika::tuple<int*, int const> ot{&value, value};

        std::vector<char> buffer;
        std::vector<pika::serialization::serialization_chunk> chunks;
        pika::serialization::output_archive oarchive(buffer, 0, &chunks);
        oarchive << ot;
        std::size_t size = oarchive.bytes_written();

        pika::serialization::input_archive iarchive(buffer, size, &chunks);
        pika::tuple<int*, int const> it{nullptr, 0};
        iarchive >> it;
        PIKA_TEST(ot == it);
    }

    // serialize raw pointer as part of std::tuple
    {
        static_assert(pika::traits::is_bitwise_serializable_v<
            pika::tuple<int*, int const>>);

        int value = 42;
        std::tuple<int*, int const> ot{&value, value};

        std::vector<char> buffer;
        std::vector<pika::serialization::serialization_chunk> chunks;
        pika::serialization::output_archive oarchive(buffer, 0, &chunks);
        oarchive << ot;
        std::size_t size = oarchive.bytes_written();

        pika::serialization::input_archive iarchive(buffer, size, &chunks);
        std::tuple<int*, int const> it{nullptr, 0};
        iarchive >> it;
        PIKA_TEST(ot == it);
    }

    // serialize tuple with a non-zero-copyable type and a const
    {
        static_assert(
            !pika::traits::is_bitwise_serializable_v<pika::tuple<A, int const>>);

        int value = 42;
        std::tuple<A, int const> ot{A(), value};

        std::vector<char> buffer;
        std::vector<pika::serialization::serialization_chunk> chunks;
        pika::serialization::output_archive oarchive(buffer, 0, &chunks);
        oarchive << ot;
        std::size_t size = oarchive.bytes_written();

        pika::serialization::input_archive iarchive(buffer, size, &chunks);
        std::tuple<A, int const> it{A(), 0};
        iarchive >> it;
        PIKA_TEST(ot == it);
    }

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    pika::local::init(pika_main, argc, argv);
    return pika::util::report_errors();
}
