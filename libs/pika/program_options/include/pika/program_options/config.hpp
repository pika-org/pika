//    Copyright (c) 2004 Hartmut Kaiser
//
//    SPDX-License-Identifier: BSL-1.0
//    Use, modification and distribution is subject to the Boost Software
//    License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//    http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/datastructures/any.hpp>
#include <pika/program_options/config/defines.hpp>

#include <optional>

namespace pika { namespace program_options {

    using any = pika::any_nonser;
    using pika::any_cast;
}}    // namespace pika::program_options
