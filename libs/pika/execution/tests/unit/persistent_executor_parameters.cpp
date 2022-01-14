//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/algorithm.hpp>
#include <pika/local/execution.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "foreach_tests.hpp"

///////////////////////////////////////////////////////////////////////////////
void test_persistent_executitor_parameters()
{
    typedef std::random_access_iterator_tag iterator_tag;
    {
        pika::execution::persistent_auto_chunk_size p;
        auto policy = pika::execution::par.with(p);
        test_for_each(policy, iterator_tag());
    }

    {
        pika::execution::persistent_auto_chunk_size p;
        auto policy = pika::execution::par(pika::execution::task).with(p);
        test_for_each_async(policy, iterator_tag());
    }

    pika::execution::parallel_executor par_exec;

    {
        pika::execution::persistent_auto_chunk_size p;
        auto policy = pika::execution::par.on(par_exec).with(p);
        test_for_each(policy, iterator_tag());
    }

    {
        pika::execution::persistent_auto_chunk_size p;
        auto policy =
            pika::execution::par(pika::execution::task).on(par_exec).with(p);
        test_for_each_async(policy, iterator_tag());
    }
}

void test_persistent_executitor_parameters_ref()
{
    using namespace pika::parallel;

    typedef std::random_access_iterator_tag iterator_tag;

    {
        pika::execution::persistent_auto_chunk_size p;
        test_for_each(pika::execution::par.with(std::ref(p)), iterator_tag());
    }

    {
        pika::execution::persistent_auto_chunk_size p;
        test_for_each_async(
            pika::execution::par(pika::execution::task).with(std::ref(p)),
            iterator_tag());
    }

    pika::execution::parallel_executor par_exec;

    {
        pika::execution::persistent_auto_chunk_size p;
        test_for_each(
            pika::execution::par.on(par_exec).with(std::ref(p)), iterator_tag());
    }

    {
        pika::execution::persistent_auto_chunk_size p;
        test_for_each_async(pika::execution::par(pika::execution::task)
                                .on(par_exec)
                                .with(std::ref(p)),
            iterator_tag());
    }
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_persistent_executitor_parameters();
    test_persistent_executitor_parameters_ref();

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace pika::program_options;
    options_description desc_commandline(
        "Usage: " PIKA_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run");

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
