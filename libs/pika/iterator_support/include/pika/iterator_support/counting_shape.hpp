//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/iterator_support/counting_iterator.hpp>
#include <pika/iterator_support/iterator_range.hpp>
#include <pika/iterator_support/range.hpp>
#include <pika/iterator_support/traits/is_range.hpp>

namespace pika { namespace util { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Incrementable>
    using counting_shape_type =
        pika::util::iterator_range<pika::util::counting_iterator<Incrementable>>;

    template <typename Incrementable>
    PIKA_HOST_DEVICE inline counting_shape_type<Incrementable>
    make_counting_shape(Incrementable n)
    {
        return pika::util::make_iterator_range(
            pika::util::make_counting_iterator(Incrementable(0)),
            pika::util::make_counting_iterator(n));
    }
}}}    // namespace pika::util::detail
