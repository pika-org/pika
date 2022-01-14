//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

namespace pika {

    /// \namespace lcos
    namespace lcos { namespace detail {
        template <typename Result>
        struct future_data;

        struct future_data_refcnt_base;
    }}    // namespace lcos::detail

    template <typename R>
    class future;

    template <typename R>
    class shared_future;

    namespace lcos {
        template <typename R>
        using future PIKA_DEPRECATED_V(
            0, 1, "pika::lcos::future is deprecated. Use pika::future instead.") =
            pika::future<R>;

        template <typename R>
        using shared_future PIKA_DEPRECATED_V(0, 1,
            "pika::lcos::shared_future is deprecated. Use pika::shared_future "
            "instead.") = pika::shared_future<R>;
    }    // namespace lcos
}    // namespace pika
