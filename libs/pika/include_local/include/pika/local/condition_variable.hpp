//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/synchronization/condition_variable.hpp>

///////////////////////////////////////////////////////////////////////////////
// C++20 condition_variable

namespace pika {

    using pika::lcos::local::condition_variable;
    using pika::lcos::local::condition_variable_any;

    using pika::lcos::local::cv_status;
}    // namespace pika
