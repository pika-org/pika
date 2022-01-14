//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/execution.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "foreach_tests.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename Parameters>
void chunk_size_test_seq(Parameters&& params)
{
    typedef std::random_access_iterator_tag iterator_tag;
    test_for_each(pika::execution::seq.with(std::ref(params)), iterator_tag());
    test_for_each_async(
        pika::execution::seq(pika::execution::task).with(std::ref(params)),
        iterator_tag());

    pika::execution::sequenced_executor seq_exec;
    test_for_each(pika::execution::seq.on(seq_exec).with(std::ref(params)),
        iterator_tag());
    test_for_each_async(pika::execution::seq(pika::execution::task)
                            .on(seq_exec)
                            .with(std::ref(params)),
        iterator_tag());
}

template <typename Parameters>
void chunk_size_test_par(Parameters&& params)
{
    typedef std::random_access_iterator_tag iterator_tag;
    test_for_each(pika::execution::par.with(std::ref(params)), iterator_tag());
    test_for_each_async(
        pika::execution::par(pika::execution::task).with(std::ref(params)),
        iterator_tag());

    pika::execution::parallel_executor par_exec;
    test_for_each(pika::execution::par.on(par_exec).with(std::ref(params)),
        iterator_tag());
    test_for_each_async(pika::execution::par(pika::execution::task)
                            .on(par_exec)
                            .with(std::ref(params)),
        iterator_tag());
}

struct timer_hooks_parameters
{
    timer_hooks_parameters(char const* name)
      : name_(name)
      , time_(0)
      , count_(0)
    {
    }

    template <typename Executor>
    void mark_begin_execution(Executor&&)
    {
        ++count_;
        time_ = pika::chrono::high_resolution_clock::now();
    }

    template <typename Executor>
    void mark_end_of_scheduling(Executor&&)
    {
    }

    template <typename Executor>
    void mark_end_execution(Executor&&)
    {
        time_ = pika::chrono::high_resolution_clock::now() - time_;
        ++count_;
    }

    std::string name_;
    std::uint64_t time_;
    std::atomic<std::size_t> count_;
};

namespace pika { namespace parallel { namespace execution {
    template <>
    struct is_executor_parameters<timer_hooks_parameters> : std::true_type
    {
    };
}}}    // namespace pika::parallel::execution

void test_timer_hooks()
{
    timer_hooks_parameters pacs("time_hooks");

    chunk_size_test_seq(pacs);
    PIKA_TEST_EQ(pacs.count_, std::size_t(8));

    chunk_size_test_par(pacs);
    PIKA_TEST_EQ(pacs.count_, std::size_t(16));
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_timer_hooks();

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
