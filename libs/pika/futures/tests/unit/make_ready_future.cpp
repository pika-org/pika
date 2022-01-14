//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <functional>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
void test_nullary_void()
{
    pika::future<void> f1 = pika::make_ready_future();
    PIKA_TEST(f1.is_ready());

    pika::future<void> f2 = pika::make_ready_future<void>();
    PIKA_TEST(f2.is_ready());
}

struct A
{
    A() = default;
};

void test_nullary()
{
    pika::future<A> f1 = pika::make_ready_future<A>();
    PIKA_TEST(f1.is_ready());
}

struct B
{
    B(int i)
      : i_(i)
    {
    }

    int i_;
};

void test_unary()
{
    B lval(42);

    pika::future<B> f1 = pika::make_ready_future(B(42));
    PIKA_TEST(f1.is_ready());
    PIKA_TEST_EQ(f1.get().i_, 42);

    pika::future<B> f2 = pika::make_ready_future(lval);
    PIKA_TEST(f2.is_ready());
    PIKA_TEST_EQ(f2.get().i_, 42);

    pika::future<B> f3 = pika::make_ready_future<B>(42);
    PIKA_TEST(f3.is_ready());
    PIKA_TEST_EQ(f3.get().i_, 42);

    pika::future<B&> f4 = pika::make_ready_future(std::ref(lval));
    PIKA_TEST(f4.is_ready());
    PIKA_TEST_EQ(&f4.get().i_, &lval.i_);

    pika::future<B&> f5 = pika::make_ready_future<B&>(lval);
    PIKA_TEST(f5.is_ready());
    PIKA_TEST_EQ(&f5.get().i_, &lval.i_);
}

struct C
{
    C(int i)
      : i_(i)
      , j_(0)
    {
    }
    C(int i, int j)
      : i_(i)
      , j_(j)
    {
    }

    C(C const&) = delete;
    C(C&& rhs)
      : i_(rhs.i_)
      , j_(rhs.j_)
    {
    }

    int i_;
    int j_;
};

void test_variadic()
{
    pika::future<C> f1 = pika::make_ready_future(C(42));
    PIKA_TEST(f1.is_ready());
    C r1 = f1.get();
    PIKA_TEST_EQ(r1.i_, 42);
    PIKA_TEST_EQ(r1.j_, 0);

    pika::future<C> f2 = pika::make_ready_future<C>(42);
    PIKA_TEST(f2.is_ready());
    C r2 = f2.get();
    PIKA_TEST_EQ(r2.i_, 42);
    PIKA_TEST_EQ(r2.j_, 0);

    pika::future<C> f3 = pika::make_ready_future(C(42, 43));
    PIKA_TEST(f3.is_ready());
    C r3 = f3.get();
    PIKA_TEST_EQ(r3.i_, 42);
    PIKA_TEST_EQ(r3.j_, 43);

    pika::future<C> f4 = pika::make_ready_future<C>(42, 43);
    PIKA_TEST(f4.is_ready());
    C r4 = f4.get();
    PIKA_TEST_EQ(r4.i_, 42);
    PIKA_TEST_EQ(r4.j_, 43);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    test_nullary_void();
    test_nullary();

    test_unary();
    test_variadic();

    return pika::local::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.cfg = cfg;

    PIKA_TEST_EQ(pika::local::init(pika_main, argc, argv, init_args), 0);
    return pika::util::report_errors();
}
