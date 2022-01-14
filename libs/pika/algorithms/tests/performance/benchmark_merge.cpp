///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <pika/local/init.hpp>
#include <pika/modules/format.hpp>
#include <pika/modules/program_options.hpp>
#include <pika/modules/testing.hpp>
#include <pika/modules/timing.hpp>
#include <pika/parallel/algorithms/generate.hpp>
#include <pika/parallel/algorithms/merge.hpp>
#include <pika/parallel/algorithms/sort.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "utils.hpp"

///////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
///////////////////////////////////////////////////////////////////////////////

struct random_fill
{
    random_fill(std::size_t random_range)
      : gen(seed)
      , dist(0, random_range - 1)
    {
    }

    int operator()()
    {
        return dist(gen);
    }

    std::mt19937 gen;
    std::uniform_int_distribution<> dist;
};

///////////////////////////////////////////////////////////////////////////////
template <typename InIter1, typename InIter2, typename OutIter>
double run_merge_benchmark_std(int test_count, InIter1 first1, InIter1 last1,
    InIter2 first2, InIter2 last2, OutIter dest)
{
    std::uint64_t time = pika::chrono::high_resolution_clock::now();

    for (int i = 0; i < test_count; ++i)
    {
        std::merge(first1, last1, first2, last2, dest);
    }

    time = pika::chrono::high_resolution_clock::now() - time;

    return (time * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
    typename FwdIter3>
double run_merge_benchmark_pika(int test_count, ExPolicy policy, FwdIter1 first1,
    FwdIter1 last1, FwdIter2 first2, FwdIter2 last2, FwdIter3 dest)
{
    std::uint64_t time = pika::chrono::high_resolution_clock::now();

    for (int i = 0; i < test_count; ++i)
    {
        pika::merge(policy, first1, last1, first2, last2, dest);
    }

    time = pika::chrono::high_resolution_clock::now() - time;

    return (time * 1e-9) / test_count;
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void run_benchmark(std::size_t vector_size1, std::size_t vector_size2,
    int test_count, std::size_t random_range, IteratorTag)
{
    std::cout << "* Preparing Benchmark..." << std::endl;

    typedef test_container<IteratorTag> test_container;
    typedef typename test_container::type container;

    container src1 = test_container::get_container(vector_size1);
    container src2 = test_container::get_container(vector_size2);
    container result =
        test_container::get_container(vector_size1 + vector_size2);

    auto first1 = std::begin(src1);
    auto last1 = std::end(src1);
    auto first2 = std::begin(src2);
    auto last2 = std::end(src2);
    auto dest = std::begin(result);

    // initialize data
    using namespace pika::execution;
    pika::generate(
        par, std::begin(src1), std::end(src1), random_fill(random_range));
    pika::generate(
        par, std::begin(src2), std::end(src2), random_fill(random_range));
    pika::sort(par, std::begin(src1), std::end(src1));
    pika::sort(par, std::begin(src2), std::end(src2));

    std::cout << "* Running Benchmark..." << std::endl;
    std::cout << "--- run_merge_benchmark_std ---" << std::endl;
    double time_std =
        run_merge_benchmark_std(test_count, first1, last1, first2, last2, dest);

    std::cout << "--- run_merge_benchmark_seq ---" << std::endl;
    double time_seq = run_merge_benchmark_pika(
        test_count, seq, first1, last1, first2, last2, dest);

    std::cout << "--- run_merge_benchmark_par ---" << std::endl;
    double time_par = run_merge_benchmark_pika(
        test_count, par, first1, last1, first2, last2, dest);

    std::cout << "--- run_merge_benchmark_par_unseq ---" << std::endl;
    double time_par_unseq = run_merge_benchmark_pika(
        test_count, par_unseq, first1, last1, first2, last2, dest);

    std::cout << "\n-------------- Benchmark Result --------------"
              << std::endl;
    auto fmt = "merge ({1}) : {2}(sec)";
    pika::util::format_to(std::cout, fmt, "std", time_std) << std::endl;
    pika::util::format_to(std::cout, fmt, "seq", time_seq) << std::endl;
    pika::util::format_to(std::cout, fmt, "par", time_par) << std::endl;
    pika::util::format_to(std::cout, fmt, "par_unseq", time_par_unseq)
        << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
std::string correct_iterator_tag_str(std::string iterator_tag)
{
    if (iterator_tag != "random"/* &&
        iterator_tag != "bidirectional" &&
        iterator_tag != "forward"*/)
        return "random";
    else
        return iterator_tag;
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    // pull values from cmd
    std::size_t vector_size = vm["vector_size"].as<std::size_t>();
    double vector_ratio = vm["vector_ratio"].as<double>();
    std::size_t random_range = vm["random_range"].as<std::size_t>();
    int test_count = vm["test_count"].as<int>();
    std::string iterator_tag_str =
        correct_iterator_tag_str(vm["iterator_tag"].as<std::string>());

    std::size_t const os_threads = pika::get_os_thread_count();

    if (random_range < 1)
        random_range = 1;

    std::size_t vector_size1 = std::size_t(vector_size * vector_ratio);
    std::size_t vector_size2 = vector_size - vector_size1;

    std::cout << "-------------- Benchmark Config --------------" << std::endl;
    std::cout << "seed         : " << seed << std::endl;
    std::cout << "vector_size1 : " << vector_size1 << std::endl;
    std::cout << "vector_size2 : " << vector_size2 << std::endl;
    std::cout << "random_range : " << random_range << std::endl;
    std::cout << "iterator_tag : " << iterator_tag_str << std::endl;
    std::cout << "test_count   : " << test_count << std::endl;
    std::cout << "os threads   : " << os_threads << std::endl;
    std::cout << "----------------------------------------------\n"
              << std::endl;

    if (iterator_tag_str == "random")
        run_benchmark(vector_size1, vector_size2, test_count, random_range,
            std::random_access_iterator_tag());
    //else if (iterator_tag_str == "bidirectional")
    //    run_benchmark(vector_size1, vector_size2, test_count, random_range,
    //        std::bidirectional_iterator_tag());
    //else // forward
    //    run_benchmark(vector_size1, vector_size2, test_count, random_range,
    //        std::forward_iterator_tag());

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    using namespace pika::program_options;
    options_description desc_commandline(
        "usage: " PIKA_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("vector_size",
        pika::program_options::value<std::size_t>()->default_value(1000000),
        "sum of sizes of two vectors (default: 1000000)")("vector_ratio",
        pika::program_options::value<double>()->default_value(0.7),
        "ratio of two vector sizes (default: 0.7)")("random_range",
        pika::program_options::value<std::size_t>()->default_value(6),
        "range of random numbers [0, x) (default: 6)")("iterator_tag",
        pika::program_options::value<std::string>()->default_value("random"),
        "the kind of iterator tag (random/bidirectional/forward)")("test_count",
        pika::program_options::value<int>()->default_value(10),
        "number of tests to be averaged (default: 10)")("seed,s",
        pika::program_options::value<unsigned int>(),
        "the random number generator seed to use for this run");

    // initialize program
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv, init_args), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
