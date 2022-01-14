//  Copyright (c) 2015-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/iterator_support/iterator_range.hpp>
#include <pika/local/execution.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "foreach_tests.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename... Parameters>
void parameters_test_impl(Parameters&&... params)
{
    static_assert(
        pika::util::all_of<
            pika::traits::is_executor_parameters<Parameters>...>::value,
        "pika::traits::is_executor_parameters<Parameters>::value");

    typedef std::random_access_iterator_tag iterator_tag;
    test_for_each(pika::execution::seq.with(params...), iterator_tag());
    test_for_each(pika::execution::par.with(params...), iterator_tag());
    test_for_each_async(
        pika::execution::seq(pika::execution::task).with(params...),
        iterator_tag());
    test_for_each_async(
        pika::execution::par(pika::execution::task).with(params...),
        iterator_tag());

    pika::execution::sequenced_executor seq_exec;
    test_for_each(
        pika::execution::seq.on(seq_exec).with(params...), iterator_tag());
    test_for_each_async(
        pika::execution::seq(pika::execution::task).on(seq_exec).with(params...),
        iterator_tag());

    pika::execution::parallel_executor par_exec;
    test_for_each(
        pika::execution::par.on(par_exec).with(params...), iterator_tag());
    test_for_each_async(
        pika::execution::par(pika::execution::task).on(par_exec).with(params...),
        iterator_tag());
}

template <typename... Parameters>
void parameters_test(Parameters&&... params)
{
    parameters_test_impl(std::ref(params)...);
    parameters_test_impl(std::forward<Parameters>(params)...);
}

void test_dynamic_chunk_size()
{
    {
        pika::execution::dynamic_chunk_size dcs;
        parameters_test(dcs);
    }

    {
        pika::execution::dynamic_chunk_size dcs(100);
        parameters_test(dcs);
    }
}

void test_static_chunk_size()
{
    {
        pika::execution::static_chunk_size scs;
        parameters_test(scs);
    }

    {
        pika::execution::static_chunk_size scs(100);
        parameters_test(scs);
    }
}

void test_guided_chunk_size()
{
    {
        pika::execution::guided_chunk_size gcs;
        parameters_test(gcs);
    }

    {
        pika::execution::guided_chunk_size gcs(100);
        parameters_test(gcs);
    }
}

void test_auto_chunk_size()
{
    {
        pika::execution::auto_chunk_size acs;
        parameters_test(acs);
    }

    {
        pika::execution::auto_chunk_size acs(std::chrono::milliseconds(1));
        parameters_test(acs);
    }
}

void test_persistent_auto_chunk_size()
{
    {
        pika::execution::persistent_auto_chunk_size pacs;
        parameters_test(pacs);
    }

    {
        pika::execution::persistent_auto_chunk_size pacs(
            std::chrono::milliseconds(0), std::chrono::milliseconds(1));
        parameters_test(pacs);
    }

    {
        pika::execution::persistent_auto_chunk_size pacs(
            std::chrono::milliseconds(0));
        parameters_test(pacs);
    }
}

///////////////////////////////////////////////////////////////////////////////
struct timer_hooks_parameters
{
    timer_hooks_parameters(char const* name)
      : name_(name)
    {
    }

    template <typename Executor>
    void mark_begin_execution(Executor&&)
    {
    }

    template <typename Executor>
    void mark_end_of_scheduling(Executor&&)
    {
    }

    template <typename Executor>
    void mark_end_execution(Executor&&)
    {
    }

    std::string name_;
};

namespace pika { namespace parallel { namespace execution {
    template <>
    struct is_executor_parameters<timer_hooks_parameters> : std::true_type
    {
    };
}}}    // namespace pika::parallel::execution

void test_combined_hooks()
{
    timer_hooks_parameters pacs("time_hooks");
    pika::execution::auto_chunk_size acs;

    parameters_test(acs, pacs);
    parameters_test(pacs, acs);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_dynamic_chunk_size();
    test_static_chunk_size();
    test_guided_chunk_size();
    test_auto_chunk_size();
    test_persistent_auto_chunk_size();

    test_combined_hooks();

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
