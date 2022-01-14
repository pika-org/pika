////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2012-2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/modules/errors.hpp>
#include <pika/topology/cpu_mask.hpp>

#include <boost/variant.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace pika { namespace threads {
    namespace detail {
        typedef std::vector<std::int64_t> bounds_type;

        enum distribution_type
        {
            compact = 0x01,
            scatter = 0x02,
            balanced = 0x04,
            numa_balanced = 0x08
        };

        using mappings_type = distribution_type;

        PIKA_EXPORT void parse_mappings(std::string const& spec,
            mappings_type& mappings, error_code& ec = throws);
    }    // namespace detail

    PIKA_EXPORT void parse_affinity_options(std::string const& spec,
        std::vector<mask_type>& affinities, std::size_t used_cores,
        std::size_t max_cores, std::size_t num_threads,
        std::vector<std::size_t>& num_pus, bool use_process_mask,
        error_code& ec = throws);

    // backwards compatibility helper
    inline void parse_affinity_options(std::string const& spec,
        std::vector<mask_type>& affinities, error_code& ec = throws)
    {
        std::vector<std::size_t> num_pus;
        parse_affinity_options(
            spec, affinities, 1, 1, affinities.size(), num_pus, false, ec);
    }
}}    // namespace pika::threads
