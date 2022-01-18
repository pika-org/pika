//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/modules/program_options.hpp>
#include <pika/modules/runtime_configuration.hpp>

#include <cstddef>

namespace pika { namespace local { namespace detail {
    PIKA_EXPORT int handle_late_commandline_options(
        util::runtime_configuration& ini,
        pika::program_options::options_description const& options,
        void (*handle_print_bind)(std::size_t));
}}}    // namespace pika::local::detail
