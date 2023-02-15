//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/future.hpp>
#include <pika/init.hpp>
#include <pika/testing.hpp>

#include <type_traits>
#include <utility>

struct test_async_executor
{
    using execution_category = pika::execution::parallel_execution_tag;

    template <typename F, typename... Ts>
    static pika::future<std::invoke_result_t<F, Ts...>>
    async_execute(F&& f, Ts&&... ts)
    {
        return pika::dataflow(
            pika::launch::async, std::forward<F>(f), std::forward<Ts>(ts)...);
    }
};

namespace pika::parallel::execution {
    template <>
    struct is_two_way_executor<test_async_executor> : std::true_type
    {
    };
}    // namespace pika::parallel::execution

int pika_main()
{
    test_async_executor exec;
    pika::future<void> f = pika::make_ready_future();
    f.then(exec, [](pika::future<void>&& f) { f.get(); });

    return pika::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return 0;
}
