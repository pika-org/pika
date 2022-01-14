//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// make inspect happy: pikainspect:nodeprecatedname:boost::is_placeholder

#pragma once

#include <pika/local/config.hpp>

#include <functional>
#include <type_traits>

namespace pika { namespace traits {
    using std::is_placeholder;
    using std::is_placeholder_v;
}}    // namespace pika::traits
