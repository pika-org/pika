//  Copyright (c)      2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>
#include <pika/pack_traversal/unwrap.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

using pika::program_options::options_description;
using pika::program_options::value;
using pika::program_options::variables_map;

using pika::dataflow;
using pika::util::bind;

using pika::async;
using pika::future;
using pika::shared_future;

using pika::make_ready_future;

using pika::local::finalize;
using pika::local::init;

using pika::unwrapping;
using pika::util::report_errors;

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

int int_f_vector(std::array<int, 10> const& vf)
{
    int sum = 0;
    for (int f : vf)
    {
        sum += f;
    }
    return sum;
}

void function_pointers()
{
    void_f_count.store(0);
    int_f_count.store(0);
    void_f1_count.store(0);
    int_f1_count.store(0);
    int_f2_count.store(0);

    future<void> f1 = dataflow(unwrapping(&void_f1), async(&int_f));
    future<int> f2 = dataflow(unwrapping(&int_f1),
        dataflow(unwrapping(&int_f1), make_ready_future(42)));
    future<int> f3 = dataflow(unwrapping(&int_f2),
        dataflow(unwrapping(&int_f1), make_ready_future(42)),
        dataflow(unwrapping(&int_f1), make_ready_future(37)));

    int_f_vector_count.store(0);
    std::array<future<int>, 10> vf;
    for (std::size_t i = 0; i < 10; ++i)
    {
        vf[i] = dataflow(unwrapping(&int_f1), make_ready_future(42));
    }
    future<int> f4 = dataflow(unwrapping(&int_f_vector), std::move(vf));

    future<int> f5 = dataflow(unwrapping(&int_f1),
        dataflow(unwrapping(&int_f1), make_ready_future(42)),
        dataflow(unwrapping(&void_f), make_ready_future()));

    f1.wait();
    PIKA_TEST_EQ(f2.get(), 126);
    PIKA_TEST_EQ(f3.get(), 163);
    PIKA_TEST_EQ(f4.get(), 10 * 84);
    PIKA_TEST_EQ(f5.get(), 126);
    PIKA_TEST_EQ(void_f_count, 1u);
    PIKA_TEST_EQ(int_f_count, 1u);
    PIKA_TEST_EQ(void_f1_count, 1u);
    PIKA_TEST_EQ(int_f1_count, 16u);
    PIKA_TEST_EQ(int_f2_count, 1u);
}

///////////////////////////////////////////////////////////////////////////////

std::atomic<std::uint32_t> future_void_f1_count;
std::atomic<std::uint32_t> future_void_f2_count;

void future_void_f1(future<void> f1)
{
    PIKA_TEST(f1.is_ready());
    ++future_void_f1_count;
}
void future_void_sf1(shared_future<void> f1)
{
    PIKA_TEST(f1.is_ready());
    ++future_void_f1_count;
}
void future_void_f2(future<void> f1, future<void> f2)
{
    PIKA_TEST(f1.is_ready());
    PIKA_TEST(f2.is_ready());
    ++future_void_f2_count;
}

std::atomic<std::uint32_t> future_int_f1_count;

int future_int_f1(future<void> f1)
{
    PIKA_TEST(f1.is_ready());
    ++future_int_f1_count;
    return 1;
}

std::atomic<std::uint32_t> future_int_f_vector_count;

int future_int_f_vector(std::array<future<int>, 10>& vf)
{
    ++future_int_f_vector_count;

    int sum = 0;
    for (future<int>& f : vf)
    {
        PIKA_TEST(f.is_ready());
        sum += f.get();
    }
    return sum;
}

void future_function_pointers()
{
    future_int_f1_count.store(0);
    future_int_f_vector_count.store(0);

    future_int_f_vector_count.store(0);
    std::array<future<int>, 10> vf;
    for (std::size_t i = 0; i < 10; ++i)
    {
        vf[i] = dataflow(&future_int_f1, make_ready_future());
    }
    future<int> f5 = dataflow(&future_int_f_vector, std::ref(vf));

    PIKA_TEST_EQ(f5.get(), 10);
    PIKA_TEST_EQ(future_int_f1_count, 10u);
    PIKA_TEST_EQ(future_int_f_vector_count, 1u);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(variables_map&)
{
    function_pointers();
    future_function_pointers();

    return pika::local::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description desc_commandline(
        "Usage: " PIKA_APPLICATION_STRING " [options]");

    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv, init_args), 0,
        "pika main exited with non-zero status");
    return report_errors();
}
