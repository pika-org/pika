////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
////////////////////////////////////////////////////////////////////////////////

#include <pika/concurrency/detail/contiguous_index_queue.hpp>
#include <pika/local/barrier.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/optional.hpp>
#include <pika/modules/testing.hpp>
#include <pika/program_options.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <random>
#include <vector>

unsigned int seed = std::random_device{}();

void test_basic()
{
    {
        // A default constructed queue should be empty.
        pika::concurrency::detail::contiguous_index_queue<> q;

        PIKA_TEST(q.empty());
        PIKA_TEST(!q.pop_left());
        PIKA_TEST(!q.pop_right());
    }

    {
        // Popping from the left should give us the expected indices.
        std::uint32_t first = 3;
        std::uint32_t last = 7;
        pika::concurrency::detail::contiguous_index_queue<> q{first, last};

        for (std::uint32_t curr_expected = first; curr_expected < last;
             ++curr_expected)
        {
            pika::optional<std::uint32_t> curr = q.pop_left();
            PIKA_TEST(curr);
            PIKA_TEST_EQ(curr.value(), curr_expected);
        }

        PIKA_TEST(q.empty());
        PIKA_TEST(!q.pop_left());
        PIKA_TEST(!q.pop_right());
    }

    {
        // Popping from the right should give us the expected indices.
        std::uint32_t first = 3;
        std::uint32_t last = 7;
        pika::concurrency::detail::contiguous_index_queue<> q{first, last};

        for (std::uint32_t curr_expected = last - 1; curr_expected >= first;
             --curr_expected)
        {
            pika::optional<std::uint32_t> curr = q.pop_right();
            PIKA_TEST(curr);
            PIKA_TEST_EQ(curr.value(), curr_expected);
        }

        PIKA_TEST(q.empty());
        PIKA_TEST(!q.pop_left());
        PIKA_TEST(!q.pop_right());

        // Resetting a queue should give us the same behaviour as a fresh
        // queue.
        q.reset(first, last);

        for (std::uint32_t curr_expected = last - 1; curr_expected >= first;
             --curr_expected)
        {
            pika::optional<std::uint32_t> curr = q.pop_right();
            PIKA_TEST(curr);
            PIKA_TEST_EQ(curr.value(), curr_expected);
        }

        PIKA_TEST(q.empty());
        PIKA_TEST(!q.pop_left());
        PIKA_TEST(!q.pop_right());
    }
}

enum class pop_mode
{
    left,
    right,
    random
};

void test_concurrent_worker(pop_mode m, std::size_t thread_index,
    pika::barrier<>& b, pika::concurrency::detail::contiguous_index_queue<>& q,
    std::vector<std::uint32_t>& popped_indices)
{
    pika::optional<std::uint32_t> curr;
    std::mt19937 r(seed + thread_index);
    std::uniform_int_distribution<> d(0, 1);

    // Make sure all threads start roughly at the same time.
    b.arrive_and_wait();

    switch (m)
    {
    case pop_mode::left:
        while ((curr = q.pop_left()))
        {
            popped_indices.push_back(curr.value());
        }
        break;
    case pop_mode::right:
        while ((curr = q.pop_right()))
        {
            popped_indices.push_back(curr.value());
        }
        break;
    case pop_mode::random:
        while (d(r) == 0 ? (curr = q.pop_left()) : (curr = q.pop_right()))
        {
            popped_indices.push_back(curr.value());
        }
        break;
    default:
        PIKA_TEST(false);
    }
}

void test_concurrent(pop_mode m)
{
    std::uint32_t first = 33;
    std::uint32_t last = 73210;
    pika::concurrency::detail::contiguous_index_queue<> q{first, last};

    std::size_t const num_threads = pika::get_num_worker_threads();
    // This test should be run on at least two worker threads.
    PIKA_TEST_LTE(std::size_t(2), num_threads);
    std::vector<pika::future<void>> fs;
    std::vector<std::vector<std::uint32_t>> popped_indices(num_threads);
    fs.reserve(num_threads);
    pika::barrier<> b(num_threads);

    for (std::size_t i = 0; i < num_threads; ++i)
    {
        fs.push_back(pika::async(test_concurrent_worker, m, i, std::ref(b),
            std::ref(q), std::ref(popped_indices[i])));
    }

    pika::wait_all(fs);

    // There should be no indices left in the queue at this point.
    PIKA_TEST(q.empty());
    PIKA_TEST(!q.pop_left());
    PIKA_TEST(!q.pop_right());

    std::size_t num_indices_expected = last - first;
    std::vector<std::uint32_t> collected_popped_indices;
    collected_popped_indices.reserve(num_indices_expected);
    std::size_t num_nonzero_indices_popped = 0;
    for (auto const& p : popped_indices)
    {
        std::copy(
            p.begin(), p.end(), std::back_inserter(collected_popped_indices));
        if (!p.empty())
        {
            ++num_nonzero_indices_popped;
        }
    }

    // All the original indices should have been popped exactly once.
    PIKA_TEST_EQ(collected_popped_indices.size(), num_indices_expected);
    std::sort(collected_popped_indices.begin(), collected_popped_indices.end());
    std::uint32_t curr_expected = first;
    for (auto const i : collected_popped_indices)
    {
        PIKA_TEST_EQ(i, curr_expected);
        ++curr_expected;
    }

    // We expect at least two threads to have popped indices concurrently.
    // There is a small chance of false positives here (resulting from big
    // delays in starting threads).
    PIKA_TEST_LTE(std::size_t(2), num_nonzero_indices_popped);
}

int pika_main(pika::program_options::variables_map& vm)
{
    if (vm.count("seed"))
    {
        seed = vm["seed"].as<unsigned int>();
    }

    test_basic();
    test_concurrent(pop_mode::left);
    test_concurrent(pop_mode::right);
    test_concurrent(pop_mode::random);
    return pika::local::finalize();
}

int main(int argc, char** argv)
{
    pika::local::init_params i;
    pika::program_options::options_description desc_cmdline(
        "usage: " PIKA_APPLICATION_STRING " [options]");
    desc_cmdline.add_options()("seed,s",
        pika::program_options::value<unsigned int>(),
        "the random number generator seed to use for this run");
    i.desc_cmdline = desc_cmdline;
    pika::local::init(pika_main, argc, argv, i);
    return pika::util::report_errors();
}
