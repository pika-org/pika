//  Copyright (c) 2013 Thomas Heller
//  Copyright (c) 2018-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/datastructures/detail/small_vector.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>
#include <pika/pack_traversal/unwrap.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace traits {

    // support unwrapping of pika::detail::small_vector
    template <typename NewType, typename OldType, std::size_t Size,
        typename OldAllocator>
    struct pack_traversal_rebind_container<NewType,
        pika::detail::small_vector<OldType, Size, OldAllocator>>
    {
        using NewAllocator = typename std::allocator_traits<
            OldAllocator>::template rebind_alloc<NewType>;

        static pika::detail::small_vector<NewType, Size, NewAllocator> call(
            pika::detail::small_vector<OldType, Size, OldAllocator> const&)
        {
            // Create a new version of the container with a new allocator
            // instance
            return pika::detail::small_vector<NewType, Size, NewAllocator>();
        }
    };
}}    // namespace pika::traits

template <typename T>
using small_vector =
    pika::detail::small_vector<T, 3, pika::util::internal_allocator<T>>;

///////////////////////////////////////////////////////////////////////////////
std::atomic<std::uint32_t> void_f_count;
std::atomic<std::uint32_t> int_f_count;

void void_f()
{
    ++void_f_count;
}
int int_f()
{
    ++int_f_count;
    return 42;
}

std::atomic<std::uint32_t> void_f1_count;
std::atomic<std::uint32_t> int_f1_count;

void void_f1(int)
{
    ++void_f1_count;
}
int int_f1(int i)
{
    ++int_f1_count;
    return i + 42;
}

std::atomic<std::uint32_t> int_f2_count;
int int_f2(int l, int r)
{
    ++int_f2_count;
    return l + r;
}

std::atomic<std::uint32_t> int_f_vector_count;

int int_f_vector(small_vector<int> const& vf)
{
    int sum = 0;
    for (int f : vf)
    {
        sum += f;
    }
    return sum;
}

void function_pointers(std::uint32_t num)
{
    void_f_count.store(0);
    int_f_count.store(0);
    void_f1_count.store(0);
    int_f1_count.store(0);
    int_f2_count.store(0);

    pika::future<void> f1 =
        pika::dataflow(pika::unwrapping(&void_f1), pika::async(&int_f));
    pika::future<int> f2 = pika::dataflow(pika::unwrapping(&int_f1),
        pika::dataflow(pika::unwrapping(&int_f1), pika::make_ready_future(42)));

    pika::future<int> f3 = pika::dataflow(pika::unwrapping(&int_f2),
        pika::dataflow(pika::unwrapping(&int_f1), pika::make_ready_future(42)),
        pika::dataflow(pika::unwrapping(&int_f1), pika::make_ready_future(37)));

    int_f_vector_count.store(0);

    small_vector<pika::future<int>> vf;
    vf.resize(num);
    for (std::uint32_t i = 0; i < num; ++i)
    {
        vf[i] =
            pika::dataflow(pika::unwrapping(&int_f1), pika::make_ready_future(42));
    }
    pika::future<int> f4 =
        pika::dataflow(pika::unwrapping(&int_f_vector), std::move(vf));

    pika::future<int> f5 = pika::dataflow(pika::unwrapping(&int_f1),
        pika::dataflow(pika::unwrapping(&int_f1), pika::make_ready_future(42)),
        pika::dataflow(pika::unwrapping(&void_f), pika::make_ready_future()));

    f1.wait();
    PIKA_TEST_EQ(f2.get(), 126);
    PIKA_TEST_EQ(f3.get(), 163);
    PIKA_TEST_EQ(f4.get(), int(num * 84));
    PIKA_TEST_EQ(f5.get(), 126);
    PIKA_TEST_EQ(void_f_count, 1u);
    PIKA_TEST_EQ(int_f_count, 1u);
    PIKA_TEST_EQ(void_f1_count, 1u);
    PIKA_TEST_EQ(int_f1_count, 6u + num);
    PIKA_TEST_EQ(int_f2_count, 1u);
}

///////////////////////////////////////////////////////////////////////////////
std::atomic<std::uint32_t> future_void_f1_count;
std::atomic<std::uint32_t> future_void_f2_count;

void future_void_f1(pika::future<void> f1)
{
    PIKA_TEST(f1.is_ready());
    ++future_void_f1_count;
}
void future_void_sf1(pika::shared_future<void> f1)
{
    PIKA_TEST(f1.is_ready());
    ++future_void_f1_count;
}
void future_void_f2(pika::future<void> f1, pika::future<void> f2)
{
    PIKA_TEST(f1.is_ready());
    PIKA_TEST(f2.is_ready());
    ++future_void_f2_count;
}

std::atomic<std::uint32_t> future_int_f1_count;

int future_int_f1(pika::future<void> f1)
{
    PIKA_TEST(f1.is_ready());
    ++future_int_f1_count;
    return 1;
}

std::atomic<std::uint32_t> future_int_f_vector_count;

int future_int_f_vector(small_vector<pika::future<int>>& vf)
{
    ++future_int_f_vector_count;

    int sum = 0;
    for (pika::future<int>& f : vf)
    {
        PIKA_TEST(f.is_ready());
        sum += f.get();
    }
    return sum;
}

void future_function_pointers(std::uint32_t num)
{
    future_int_f1_count.store(0);
    future_int_f_vector_count.store(0);

    future_int_f_vector_count.store(0);
    small_vector<pika::future<int>> vf;
    vf.resize(num);
    for (std::uint32_t i = 0; i < num; ++i)
    {
        vf[i] = pika::dataflow(&future_int_f1, pika::make_ready_future());
    }
    pika::future<int> f5 = pika::dataflow(&future_int_f_vector, std::ref(vf));

    PIKA_TEST_EQ(f5.get(), int(num));
    PIKA_TEST_EQ(future_int_f1_count, num);
    PIKA_TEST_EQ(future_int_f_vector_count, 1u);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    function_pointers(3);
    function_pointers(3);
    future_function_pointers(10);
    future_function_pointers(10);

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
