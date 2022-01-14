//  Copyright (c) 2021 Srinivas Yadav
//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/parallel/datapar.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "../algorithms/copy_tests.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_copy()
{
    using namespace pika::execution;

    test_copy(IteratorTag());

    test_copy(simd, IteratorTag());
    test_copy(par_simd, IteratorTag());

    test_copy_async(simd(task), IteratorTag());
    test_copy_async(par_simd(task), IteratorTag());
}

void copy_test()
{
    test_copy<std::random_access_iterator_tag>();
    test_copy<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_copy_exception()
{
    using namespace pika::execution;

    test_copy_exception(IteratorTag());

    test_copy_exception(simd, IteratorTag());
    test_copy_exception(par_simd, IteratorTag());

    test_copy_exception_async(simd(task), IteratorTag());
    test_copy_exception_async(par_simd(task), IteratorTag());
}

void copy_exception_test()
{
    test_copy_exception<std::random_access_iterator_tag>();
    test_copy_exception<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_copy_bad_alloc()
{
    test_copy_bad_alloc(pika::execution::simd, IteratorTag());
    test_copy_bad_alloc(pika::execution::par_simd, IteratorTag());

    test_copy_bad_alloc_async(
        pika::execution::simd(pika::execution::task), IteratorTag());
    test_copy_bad_alloc_async(
        pika::execution::par_simd(pika::execution::task), IteratorTag());
}

void copy_bad_alloc_test()
{
    test_copy_bad_alloc<std::random_access_iterator_tag>();
    test_copy_bad_alloc<std::forward_iterator_tag>();
}

int pika_main(pika::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    copy_test();
    copy_exception_test();
    copy_bad_alloc_test();
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
