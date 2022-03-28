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

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/errors/error_code.hpp>
#include <pika/topology/cpu_mask.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pika::detail {
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

    PIKA_EXPORT void parse_affinity_options(std::string const& spec,
        std::vector<threads::mask_type>& affinities, std::size_t used_cores,
        std::size_t max_cores, std::size_t num_threads,
        std::vector<std::size_t>& num_pus, bool use_process_mask,
        error_code& ec = throws);
}    // namespace pika::detail
