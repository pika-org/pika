//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>

#include <pika/local/algorithm.hpp>
#include <pika/local/execution.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    std::vector<int> v(100);

    {
        pika::execution::static_chunk_size block(1);
        pika::execution::parallel_executor exec;
        pika::ranges::for_each(
            pika::execution::par.on(exec).with(block), v, [](int) {});
    }

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    return pika::local::init(pika_main, argc, argv);
}
