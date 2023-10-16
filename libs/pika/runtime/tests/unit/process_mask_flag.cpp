//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test checks that the runtime correctly converts physical pu indices to logical ones when
// setting the process mask. This test assumes that the system has two hardware threads per core.
//
// The test sets an explicit mask with --pika:process-mask and prints the resulting bindings with
// --pika:print-bind. The test runner checks that the output is as expected.
//
// We set the mask to 0x3 (i.e. two first bits set). This should result in two threads being created
// by the runtime as the first to physical hardware threads are on separate cores. If the mask is
// interpreted as logical indices the runtime will only start use one thread, as hardware threads
// within a core are consecutive with logical indices.

#include <pika/init.hpp>
#include <pika/testing.hpp>

int main(int argc, char** argv)
{
    pika::init_params init_args;
    init_args.cfg = {"--pika:process-mask=0x3", "--pika:print-bind"};

    pika::start(nullptr, argc, argv, init_args);
    pika::finalize();
    pika::stop();

    PIKA_TEST(true);

    return 0;
}
