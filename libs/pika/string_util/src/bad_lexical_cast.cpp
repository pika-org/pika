//  Copyright (c) 2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if defined(PIKA_HAVE_MODULE)
module;
#endif

#if !defined(PIKA_HAVE_MODULE)
#include <pika/string_util/bad_lexical_cast.hpp>
#include <typeinfo>
#endif

#if defined(PIKA_HAVE_MODULE)
module pika.string_util;
#endif

namespace pika::detail {
    const char* bad_lexical_cast::what() const noexcept
    {
        return "bad lexical cast: source type value could not be interpreted as target";
    }

    bad_lexical_cast::~bad_lexical_cast() noexcept = default;

    void throw_bad_lexical_cast(
        std::type_info const& source_type, std::type_info const& target_type)
    {
        throw bad_lexical_cast(source_type, target_type);
    }
}    // namespace pika::detail
