//  Copyright (c) 2019 Jeff Trull
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/algorithm.hpp>
#include <pika/local/init.hpp>
#include <pika/local/numeric.hpp>
#include <pika/modules/testing.hpp>

#include <cstddef>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

struct test_async_executor : pika::execution::parallel_executor
{
    template <typename F, typename T, typename... Ts>
    pika::future<
        typename pika::util::invoke_result<F, T, std::size_t, Ts...>::type>
    async_execute(F&& f, T&& t, std::size_t chunk_size, Ts&&... ts)
    {
        // make sure the chunk_size is equal to what was specified below
        PIKA_TEST_EQ(chunk_size, std::size_t(50000));

        using base_type = pika::execution::parallel_executor;
        return this->base_type::async_execute(std::forward<F>(f),
            std::forward<T>(t), chunk_size, std::forward<Ts>(ts)...);
    }
};

namespace pika { namespace parallel { namespace execution {
    template <>
    struct is_two_way_executor<test_async_executor> : std::true_type
    {
    };
}}}    // namespace pika::parallel::execution

int pika_main()
{
    using namespace pika::util;

    // create a fixed chunk size to be used in the algorithm
    pika::execution::static_chunk_size fixed(50000);

    // helper-executor to verify the used chunk-size
    test_async_executor exec;

    // this does not seem to be obeyed!
    auto ex = pika::execution::par.on(exec).with(fixed);

    // create and fill random vector of desired size
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::uniform_int_distribution<int> dist{1, 20};

    std::size_t sz = 16750000;

    std::vector<int> data(sz);
    std::generate(
        data.begin(), data.end(), [&]() { return dist(mersenne_engine); });

    std::vector<int> result(sz + 1);

    pika::exclusive_scan(ex, data.begin(), data.end(), result.begin(), 0);

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    // By default this should run on all available cores
    std::vector<std::string> const cfg = {"pika.os_threads=2"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.cfg = cfg;

    PIKA_TEST_EQ(pika::local::init(pika_main, argc, argv, init_args), 0);

    return pika::util::report_errors();
}
