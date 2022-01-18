//  Copyright (c)      2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/functional/bind_back.hpp>
#include <pika/functional/function.hpp>
#include <pika/ini/ini.hpp>
#include <pika/prefix/find_prefix.hpp>
#include <pika/resource_partitioner/partitioner_fwd.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace pika { namespace resource { namespace detail {
    PIKA_EXPORT partitioner& create_partitioner(
        resource::partitioner_mode rpmode, pika::util::section rtcfg,
        pika::threads::policies::detail::affinity_data affinity_data);

}}}    // namespace pika::resource::detail
