//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

module;
// // #include <pika/assertion/source_location.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <ostream>
#endif

module pika.assertion;

import std;

namespace pika::detail {
    std::ostream& operator<<(std::ostream& os, source_location const& loc)
    {
        os << loc.file_name << ":" << loc.line_number << ": " << loc.function_name;
        return os;
    }
}    // namespace pika::detail
