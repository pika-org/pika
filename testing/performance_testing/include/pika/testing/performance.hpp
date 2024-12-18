//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/functional/function.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

namespace pika::util {

    namespace detail {
        // Json output for performance reports
        class json_perf_times
        {
            using key_t = std::tuple<std::string>;
            using value_t = std::vector<double>;
            using map_t = std::map<key_t, value_t>;

            map_t m_map;

            friend std::ostream& operator<<(std::ostream& strm, json_perf_times const& obj)
            {
                strm << "{\n";
                strm << "  \"outputs\" : [";
                int outputs = 0;
                for (auto&& item : obj.m_map)
                {
                    if (outputs) strm << ",";
                    strm << "\n    {\n";
                    strm << "      \"name\" : \"" << std::get<0>(item.first) << "\",\n";
                    strm << "      \"series\" : [";
                    int series = 0;
                    for (auto val : item.second)
                    {
                        if (series) strm << ", ";
                        strm << val;
                        ++series;
                    }
                    strm << "]\n";
                    strm << "    }";
                    ++outputs;
                }
                if (outputs) strm << "\n  ";
                strm << "]\n";
                strm << "}\n";
                return strm;
            }

        public:
            void add(std::string const& name, double time) { m_map[key_t(name)].push_back(time); }
        };

        json_perf_times& times();

        // Add time to the map for performance report
        void add_time(std::string const& test_name, double time);
    }    // namespace detail

    PIKA_EXPORT void perftests_report(
        std::string const& name, std::size_t const steps, detail::function<void(void)>&& test);

    PIKA_EXPORT void perftests_print_times();

    PIKA_EXPORT void print_cdash_timing(char const* name, double time);
    PIKA_EXPORT void print_cdash_timing(char const* name, std::uint64_t time);
}    // namespace pika::util
