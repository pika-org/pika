//  Copyright (c) 2020 albestro
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <atomic>
#include <string>
#include <vector>

std::atomic<bool> main_executed(false);

int pika_main()
{
    main_executed = true;
    return pika::local::finalize();
}

int main(int argc, char** argv)
{
    std::vector<std::string> cfg = {"--pika:help"};

    pika::local::init_params init_args;
    init_args.cfg = cfg;

    PIKA_TEST_EQ(pika::local::init(pika_main, argc, argv, init_args), 0);

    PIKA_TEST(!main_executed);

    return pika::util::report_errors();
}
