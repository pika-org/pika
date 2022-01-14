//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/futures/traits/future_traits.hpp>
#include <pika/type_support/always_void.hpp>

#include <iterator>

namespace pika { namespace lcos { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename Enable = void>
    struct future_iterator_traits
    {
    };

    template <typename Iterator>
    struct future_iterator_traits<Iterator,
        pika::util::always_void_t<
            typename std::iterator_traits<Iterator>::value_type>>
    {
        using type = typename std::iterator_traits<Iterator>::value_type;
        using traits_type = pika::traits::future_traits<type>;
    };

    template <typename Iter>
    using future_iterator_traits_t =
        typename future_iterator_traits<Iter>::type;
}}}    // namespace pika::lcos::detail
