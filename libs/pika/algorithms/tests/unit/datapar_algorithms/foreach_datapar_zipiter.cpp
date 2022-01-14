//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/algorithms/for_each.hpp>
#include <pika/parallel/datapar.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <pika/modules/program_options.hpp>

#include "../algorithms/test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
struct set_42
{
    template <typename Tuple>
    void operator()(Tuple&& t)
    {
        pika::get<0>(t) = 42;
        pika::get<1>(t) = 42;
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void for_each_zipiter_test(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007), d(10007);
    std::iota(std::begin(c), std::end(c), std::rand());
    std::iota(std::begin(d), std::end(d), std::rand());

    auto begin = pika::util::make_zip_iterator(
        iterator(std::begin(c)), iterator(std::begin(d)));
    auto end = pika::util::make_zip_iterator(
        iterator(std::end(c)), iterator(std::end(d)));

    // auto result =
    pika::for_each(std::forward<ExPolicy>(policy), begin, end, set_42());

    // PIKA_TEST_EQ(result, end);

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](int v) -> void {
        PIKA_TEST_EQ(v, int(42));
        ++count;
    });
    PIKA_TEST_EQ(count, c.size());
    /*
    auto begin = pika::util::make_zip_iterator(
        iterator(std::begin(c)), iterator(std::begin(d)));

    static_assert(
//        pika::parallel::traits::is_indirect_callable<
//            set_42, pika::parallel::traits::projected<
//                pika::parallel::util::projection_identity,
//                decltype(begin)>
//        >::value,
//        pika::is_invocable_v<
//            set_42, typename std::iterator_traits<decltype(begin)>::value_type&,
//            typename pika::parallel::traits::detail::projected_result_of_indirect<
//            pika::parallel::traits::projected<
//                pika::parallel::util::projection_identity,
//                decltype(begin)
//            >
//            >::type)
        >,
        "foo");*/
}

template <typename IteratorTag>
void for_each_zipiter_test()
{
    using namespace pika::execution;

    for_each_zipiter_test(par_simd, IteratorTag());
    //     test_for_each_async(par_simd(task), IteratorTag());
}

void for_each_zipiter_test()
{
    for_each_zipiter_test<std::random_access_iterator_tag>();
    //    for_each_zipiter_test<std::forward_iterator_tag>();
    //    for_each_zipiter_test<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    for_each_zipiter_test();

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
