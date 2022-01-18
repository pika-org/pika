//  Copyright (c) 2021 Nanmiao Wu
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int pika_main(int argc, char* argv[])
{
    PIKA_TEST_EQ(argc, 2);
    PIKA_TEST_EQ(std::string(argv[1]), std::string("-wobble=1"));

    return pika::local::finalize();
}
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // pass unknown command line option that would conflict with predefined
    // alias (-w)
    std::vector<std::string> const cfg = {
        "--pika:ini=pika.commandline.allow_unknown!=1",
        "--pika:ini=pika.commandline.aliasing!=0"};

    pika::local::init_params init_args;
    init_args.cfg = cfg;

    PIKA_TEST_EQ(pika::local::init(pika_main, argc, argv, init_args), 0);

    return pika::util::report_errors();
}
