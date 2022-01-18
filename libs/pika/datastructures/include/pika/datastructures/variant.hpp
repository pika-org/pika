//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

#if defined(PIKA_HAVE_CXX17_COPY_ELISION)
#include <variant>

namespace pika {

    using std::get;
    using std::holds_alternative;
    using std::monostate;
    using std::variant;
    using std::visit;
}    // namespace pika

#else

#include <pika/datastructures/detail/variant.hpp>

namespace pika {

    using pika::variant_ns::get;
    using pika::variant_ns::holds_alternative;
    using pika::variant_ns::monostate;
    using pika::variant_ns::variant;
    using pika::variant_ns::visit;
}    // namespace pika

#endif
