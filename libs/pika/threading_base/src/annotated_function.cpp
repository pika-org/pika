//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>

#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
#include <pika/threading_base/annotated_function.hpp>

#include <string>
#include <unordered_set>
#include <utility>

namespace pika::detail {
    char const* store_function_annotation(std::string name)
    {
        static thread_local std::unordered_set<std::string> names;
        auto r = names.emplace(PIKA_MOVE(name));
        return (*std::get<0>(r)).c_str();
    }
}    // namespace pika::detail

#else

#include <string>

namespace pika::detail {
    char const* store_function_annotation(std::string)
    {
        return "<unknown>";
    }
}    // namespace pika::detail

#endif
