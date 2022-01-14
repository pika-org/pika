//  Copyright (c) 2015 Daniel Bourgeois
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/algorithm.hpp>
#include <pika/local/execution.hpp>
#include <pika/local/init.hpp>

#include <type_traits>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct my_executor : pika::execution::parallel_executor
{
};

namespace pika { namespace parallel { namespace execution {
    template <>
    struct is_one_way_executor<my_executor> : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<my_executor> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<my_executor> : std::true_type
    {
    };
}}}    // namespace pika::parallel::execution

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    my_executor exec;

    std::vector<int> v(100);

    pika::ranges::for_each(pika::execution::par.on(exec), v, [](int) {});

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    return pika::local::init(pika_main, argc, argv);
}
