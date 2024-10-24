//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/testing/performance.hpp>

#include <fmt/ostream.h>
#include <fmt/printf.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>

namespace pika::util {

    namespace detail {

        json_perf_times& times()
        {
            static json_perf_times res;
            return res;
        }

        void add_time(std::string const& test_name, double time) { times().add(test_name, time); }

    }    // namespace detail

    void perftests_report(
        std::string const& name, const std::size_t steps, detail::function<void(void)>&& test)
    {
        if (steps == 0) return;
        // First iteration to cache the data
        test();
        using timer = std::chrono::high_resolution_clock;
        timer::time_point start;
        for (size_t i = 0; i != steps; ++i)
        {
            // For now we don't flush the cache
            //flush_cache();
            start = timer::now();
            test();
            // default is in seconds
            auto time =
                std::chrono::duration_cast<std::chrono::duration<double>>(timer::now() - start);
            detail::add_time(name, time.count());
        }
    }

    void perftests_print_times() { std::cout << detail::times(); }

    void print_cdash_timing(char const* name, double time)
    {
        fmt::print(std::cout,
            "<DartMeasurement name=\"{}\" type=\"numeric/double\">{}</DartMeasurement>\n", name,
            time);
    }

    void print_cdash_timing(char const* name, std::uint64_t time)
    {
        print_cdash_timing(name, static_cast<double>(time) / 1e9);
    }
}    // namespace pika::util
