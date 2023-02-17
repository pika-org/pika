//  Copyright (c) 2018 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/future.hpp>
#include <pika/init.hpp>

#include <iostream>
#include <random>
#include <tuple>
#include <vector>

void final_task(pika::future<std::tuple<pika::future<double>, pika::future<void>>>)
{
    std::cout << "in final_task" << std::endl;
}

int pika_main()
{
    // A function can be launched asynchronously. The program will not block
    // here until the result is available.
    pika::future<int> f = pika::async([]() { return 42; });
    std::cout << "Just launched a task!" << std::endl;

    // Use get to retrieve the value from the future. This will block this task
    // until the future is ready, but the pika runtime will schedule other tasks
    // if there are tasks available.
    std::cout << "f contains " << f.get() << std::endl;

    // Let's launch another task.
    pika::future<double> g = pika::async([]() { return 3.14; });

    // Tasks can be chained using the then method. The continuation takes the
    // future as an argument.
    pika::future<double> result = g.then([](pika::future<double>&& gg) {
        // This function will be called once g is ready. gg is g moved
        // into the continuation.
        return gg.get() * 42.0 * 42.0;
    });

    // You can check if a future is ready with the is_ready method.
    std::cout << "Result is ready? " << result.is_ready() << std::endl;

    // You can launch other work in the meantime. Let's start an intermediate
    // task that prints to std::cout.
    pika::future<void> intermediate_task =
        pika::async([]() { std::cout << "hello from intermediate task" << std::endl; });

    // We launch the final task when the intermediate task is ready and result
    // is ready using when_all.
    auto all = pika::when_all(result, intermediate_task).then(&final_task);

    // We can wait for all to be ready.
    all.wait();

    // all must be ready at this point because we waited for it to be ready.
    std::cout << (all.is_ready() ? "all is ready!" : "all is not ready...") << std::endl;

    return pika::finalize();
}

int main(int argc, char* argv[])
{
    return pika::init(pika_main, argc, argv);
}
