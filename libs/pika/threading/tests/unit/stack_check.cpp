// Copyright (C) 2016 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/coroutines/detail/get_stack_pointer.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

typedef std::tuple<std::size_t, std::ptrdiff_t, std::ptrdiff_t> info;
typedef std::stack<info> info_stack;

void stack_remaining(const char* txt, info_stack& stack)
{
#if defined(PIKA_HAVE_THREADS_GET_STACK_POINTER)
    std::size_t stack_ptr = pika::threads::coroutines::detail::get_stack_ptr();
#else
    std::size_t stack_ptr = 0x00000000;
#endif
    std::ptrdiff_t stacksize = pika::this_thread::get_stack_size();
    std::ptrdiff_t remaining_stack =
        pika::this_thread::get_available_stack_space();
    //
    std::cout << txt << " stacksize       : 0x" << std::hex << stacksize
              << "\n";
    std::cout << txt << " stack pointer   : 0x" << std::hex << stack_ptr
              << "\n";
    std::cout << txt << " stack remaining : 0x" << std::hex << remaining_stack
              << "\n\n";

    stack.push(std::make_tuple(stack_ptr, stacksize, remaining_stack));
}
//

void stack_waste(int N, info_stack& stack)
{
    // declare 1 MB of stack vars
    char bytes[1 << 10] = {0};
    // prevent the compiler optimizing it away
    std::fill_n(&bytes[0], 32, 0);
    std::stringstream dummy;
    dummy << bytes[45] << std::ends;
    //
    std::stringstream temp;
    temp << "stack_waste " << N;
    stack_remaining(temp.str().c_str(), stack);
    //
    if (N > 0)
        stack_waste(N - 1, stack);
}
//

int pika_main()
{
    info_stack my_stack_info;
    // just for curiosity
    stack_remaining("pika_main", my_stack_info);

    // test stack vars
    stack_waste(20, my_stack_info);

    std::ptrdiff_t current_stack = 0;
    while (!my_stack_info.empty())
    {
        info i = my_stack_info.top();
        std::ptrdiff_t stack_now = std::get<2>(i);
        std::cout << "stack remaining 0x" << std::hex << stack_now << "\n";
#if defined(PIKA_HAVE_THREADS_GET_STACK_POINTER)
        PIKA_TEST_LT(current_stack, stack_now);
#endif
        current_stack = stack_now;
        my_stack_info.pop();
    }
    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    //
    // add command line option which controls the random number generator seed
    using namespace pika::program_options;
    options_description desc_commandline(
        "Usage: " PIKA_APPLICATION_STRING " [options]");

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv, init_args), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
