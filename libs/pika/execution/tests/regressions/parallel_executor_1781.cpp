//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/init.hpp>

#include <pika/algorithm.hpp>
#include <pika/execution.hpp>

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

    return pika::finalize();
}

int main(int argc, char* argv[])
{
    return pika::init(pika_main, argc, argv);
}
