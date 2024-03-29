//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/functional/function.hpp>
#include <pika/functional/unique_function.hpp>

namespace pika::util::detail {
    template <typename Sig>
    inline void reset_function(pika::util::detail::function<Sig>& f)
    {
        f.reset();
    }

    template <typename Sig>
    inline void reset_function(pika::util::detail::unique_function<Sig>& f)
    {
        f.reset();
    }

    template <typename Function>
    inline void reset_function(Function& f)
    {
        f = Function();
    }
}    // namespace pika::util::detail
