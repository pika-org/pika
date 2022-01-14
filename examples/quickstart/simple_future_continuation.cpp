//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/execution.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/runtime.hpp>
#include <pika/local/thread.hpp>

#include <iostream>

int get_id(int i)
{
    return i;
}

int func1()
{
    std::cout << "func1 thread id: " << pika::this_thread::get_id() << std::endl;
    return get_id(1) ? 123 : 0;
}

// this continuation function will be executed by an pika thread
int cont1(pika::future<int> f)
{
    std::cout << "cont1 thread id: " << pika::this_thread::get_id() << std::endl;
    std::cout << "Status code (pika thread): " << f.get() << std::endl;
    std::cout << std::flush;
    return 1;
}

// this continuation function will be executed by the UI (main) thread, which is
// not an pika thread
int cont2(pika::future<int> f)
{
    std::cout << "Status code (main thread): " << f.get() << std::endl;
    return 1;
}

int pika_main()
{
    // executing continuation cont1 on same thread as func1
    {
        pika::future<int> t = pika::async(&func1);
        pika::future<int> t2 = t.then(pika::launch::sync, &cont1);
        t2.get();
    }

    // executing continuation cont1 on new pika thread
    {
        pika::future<int> t = pika::async(&func1);
        pika::future<int> t2 = t.then(pika::launch::async, &cont1);
        t2.get();
    }

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    return pika::local::init(pika_main, argc, argv);
}
