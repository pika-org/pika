//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/algorithm.hpp>
#include <pika/local/chrono.hpp>
#include <pika/local/execution.hpp>
#include <pika/local/init.hpp>
#include <pika/local/numeric.hpp>

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct plus
{
    template <typename T1, typename T2>
    auto operator()(T1&& t1, T2&& t2) const -> decltype(t1 + t2)
    {
        return t1 + t2;
    }
};

struct multiplies
{
    template <typename T1, typename T2>
    auto operator()(T1&& t1, T2&& t2) const -> decltype(t1 * t2)
    {
        return t1 * t2;
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
float measure_inner_product(ExPolicy&& policy, std::vector<float> const& data1,
    std::vector<float> const& data2)
{
    return pika::transform_reduce(policy, std::begin(data1), std::end(data1),
        std::begin(data2), 0.0f, ::multiplies(), ::plus());
}

template <typename ExPolicy>
std::int64_t measure_inner_product(int count, ExPolicy&& policy,
    std::vector<float> const& data1, std::vector<float> const& data2)
{
    std::int64_t start = pika::chrono::high_resolution_clock::now();

    for (int i = 0; i != count; ++i)
        measure_inner_product(policy, data1, data2);

    return (pika::chrono::high_resolution_clock::now() - start) / count;
}

int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::random_device{}();
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::mt19937 gen(seed);

    std::size_t size = vm["vector_size"].as<std::size_t>();
    bool csvoutput = vm["csv_output"].as<int>() ? true : false;
    int test_count = vm["test_count"].as<int>();

    std::vector<float> data1(size);
    std::vector<float> data2(size);

    std::iota(std::begin(data1), std::end(data1), float(gen()));
    std::iota(std::begin(data2), std::end(data2), float(gen()));

    if (test_count <= 0)
    {
        std::cout << "test_count cannot be less than zero...\n" << std::flush;
    }
    else
    {
        // warm up caches
        measure_inner_product(pika::execution::par, data1, data2);

        // do measurements
        std::uint64_t tr_time_datapar = measure_inner_product(
            test_count, pika::execution::par_simd, data1, data2);
        std::uint64_t tr_time_par = measure_inner_product(
            test_count, pika::execution::par, data1, data2);

        if (csvoutput)
        {
            std::cout << "," << tr_time_par / 1e9 << ","
                      << tr_time_datapar / 1e9 << "\n"
                      << std::flush;
        }
        else
        {
            std::cout << "transform_reduce(execution::par): " << std::right
                      << std::setw(15) << tr_time_par / 1e9 << "\n"
                      << "transform_reduce(datapar): " << std::right
                      << std::setw(15) << tr_time_datapar / 1e9 << "\n"
                      << std::flush;
        }
    }

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    pika::program_options::options_description cmdline(
        "usage: " PIKA_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()
        ("vector_size"
        , pika::program_options::value<std::size_t>()->default_value(1024)
        , "size of vector")

        ("csv_output"
        , pika::program_options::value<int>()->default_value(0)
        , "print results in csv format")

        ("test_count"
        , pika::program_options::value<int>()->default_value(10)
        , "number of tests to take average from")

        ("seed,s"
        , pika::program_options::value<unsigned int>()
        , "the random number generator seed to use for this run")
        ;
    // clang-format on

    pika::local::init_params init_args;
    init_args.desc_cmdline = cmdline;
    init_args.cfg = cfg;

    return pika::local::init(pika_main, argc, argv, init_args);
}
