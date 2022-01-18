//  Copyright (c)      2020 ETH Zurich
//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/local/execution.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
struct additional_argument
{
};

struct additional_argument_executor
{
    template <typename F, typename... Ts,
        typename Enable = typename std::enable_if<
            !std::is_member_function_pointer<F>::value>::type>
    decltype(auto) async_execute(F&& f, Ts&&... ts)
    {
        return pika::async(
            std::forward<F>(f), additional_argument{}, std::forward<Ts>(ts)...);
    }

    template <typename F, typename T, typename... Ts,
        typename Enable = typename std::enable_if<
            std::is_member_function_pointer<F>::value>::type>
    decltype(auto) async_execute(F&& f, T&& t, Ts&&... ts)
    {
        return pika::async(std::forward<F>(f), std::forward<T>(t),
            additional_argument{}, std::forward<Ts>(ts)...);
    }
};

namespace pika { namespace parallel { namespace execution {
    template <>
    struct is_two_way_executor<additional_argument_executor> : std::true_type
    {
    };
}}}    // namespace pika::parallel::execution

///////////////////////////////////////////////////////////////////////////////
std::int32_t increment(additional_argument, std::int32_t i)
{
    return i + 1;
}

std::int32_t increment_with_future(
    additional_argument, pika::shared_future<std::int32_t> fi)
{
    return fi.get() + 1;
}

///////////////////////////////////////////////////////////////////////////////
struct mult2
{
    std::int32_t operator()(additional_argument, std::int32_t i) const
    {
        return i * 2;
    }
};

///////////////////////////////////////////////////////////////////////////////
struct decrement
{
    std::int32_t call(additional_argument, std::int32_t i) const
    {
        return i - 1;
    }
};

///////////////////////////////////////////////////////////////////////////////
void do_nothing(additional_argument, std::int32_t) {}

struct do_nothing_obj
{
    void operator()(additional_argument, std::int32_t) const {}
};

struct do_nothing_member
{
    void call(additional_argument, std::int32_t) const {}
};

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_async_with_executor(Executor& exec)
{
    {
        pika::future<std::int32_t> f1 = pika::async(exec, &increment, 42);
        PIKA_TEST_EQ(f1.get(), 43);

        pika::future<void> f2 = pika::async(exec, &do_nothing, 42);
        f2.get();
    }

    {
        pika::lcos::local::promise<std::int32_t> p;
        pika::shared_future<std::int32_t> f = p.get_future();

        pika::future<std::int32_t> f1 =
            pika::async(exec, &increment_with_future, f);
        pika::future<std::int32_t> f2 =
            pika::async(exec, &increment_with_future, f);

        p.set_value(42);
        PIKA_TEST_EQ(f1.get(), 43);
        PIKA_TEST_EQ(f2.get(), 43);
    }

    {
        using pika::util::placeholders::_1;
        using pika::util::placeholders::_2;

        pika::future<std::int32_t> f1 =
            pika::async(exec, pika::util::bind_back(&increment, 42));
        PIKA_TEST_EQ(f1.get(), 43);

        pika::future<std::int32_t> f2 =
            pika::async(exec, pika::util::bind(&increment, _1, _2), 42);
        PIKA_TEST_EQ(f2.get(), 43);
    }

    {
        pika::future<std::int32_t> f1 = pika::async(exec, increment, 42);
        PIKA_TEST_EQ(f1.get(), 43);

        pika::future<void> f2 = pika::async(exec, do_nothing, 42);
        f2.get();
    }

    {
        mult2 mult;

        pika::future<std::int32_t> f1 = pika::async(exec, mult, 42);
        PIKA_TEST_EQ(f1.get(), 84);
    }

    {
        mult2 mult;

        pika::future<std::int32_t> f1 =
            pika::async(exec, pika::util::bind_back(mult, 42));
        PIKA_TEST_EQ(f1.get(), 84);

        using pika::util::placeholders::_1;
        using pika::util::placeholders::_2;

        pika::future<std::int32_t> f2 =
            pika::async(exec, pika::util::bind(mult, _1, _2), 42);
        PIKA_TEST_EQ(f2.get(), 84);

        do_nothing_obj do_nothing_f;
        pika::future<void> f3 =
            pika::async(exec, pika::util::bind(do_nothing_f, _1, _2), 42);
        f3.get();
    }

    {
        decrement dec;

        pika::future<std::int32_t> f1 =
            pika::async(exec, &decrement::call, dec, 42);
        PIKA_TEST_EQ(f1.get(), 41);

        do_nothing_member dnm;
        pika::future<void> f2 =
            pika::async(exec, &do_nothing_member::call, dnm, 42);
        f2.get();
    }

    {
        decrement dec;

        using pika::util::placeholders::_1;
        using pika::util::placeholders::_2;

        pika::future<std::int32_t> f1 =
            pika::async(exec, pika::util::bind(&decrement::call, dec, _1, 42));
        PIKA_TEST_EQ(f1.get(), 41);

        pika::future<std::int32_t> f2 = pika::async(
            exec, pika::util::bind(&decrement::call, dec, _1, _2), 42);
        PIKA_TEST_EQ(f2.get(), 41);

        do_nothing_member dnm;
        pika::future<void> f3 = pika::async(
            exec, pika::util::bind(&do_nothing_member::call, dnm, _1, _2), 42);
        f3.get();
    }
}

int pika_main()
{
    {
        additional_argument_executor exec;
        test_async_with_executor(exec);
    }

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run pika
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
