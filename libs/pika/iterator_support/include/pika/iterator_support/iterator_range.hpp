//  Copyright (c) 2017 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/iterator_support/range.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/iterator_support/traits/is_range.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace pika::util {
    template <typename Iterator, typename Sentinel = Iterator>
    class iterator_range
    {
    public:
        iterator_range() = default;

        PIKA_HOST_DEVICE iterator_range(Iterator iterator, Sentinel sentinel)
          : _iterator(std::move(iterator))
          , _sentinel(std::move(sentinel))
        {
        }

        PIKA_HOST_DEVICE Iterator begin() const { return _iterator; }

        PIKA_HOST_DEVICE Iterator end() const { return _sentinel; }

        PIKA_HOST_DEVICE std::ptrdiff_t size() const { return std::distance(_iterator, _sentinel); }

        PIKA_HOST_DEVICE bool empty() const { return _iterator == _sentinel; }

    private:
        Iterator _iterator;
        Sentinel _sentinel;
    };

    template <typename Range, typename Iterator = typename traits::range_iterator<Range>::type,
        typename Sentinel = typename traits::range_iterator<Range>::type>
    typename std::enable_if<traits::is_range<Range>::value,
        iterator_range<Iterator, Sentinel>>::type
    make_iterator_range(Range& r)
    {
        return iterator_range<Iterator, Sentinel>(util::begin(r), util::end(r));
    }

    template <typename Range,
        typename Iterator = typename traits::range_iterator<Range const>::type,
        typename Sentinel = typename traits::range_iterator<Range const>::type>
    typename std::enable_if<traits::is_range<Range>::value,
        iterator_range<Iterator, Sentinel>>::type
    make_iterator_range(Range const& r)
    {
        return iterator_range<Iterator, Sentinel>(util::begin(r), util::end(r));
    }

    template <typename Iterator, typename Sentinel = Iterator>
    typename std::enable_if<traits::is_iterator<Iterator>::value,
        iterator_range<Iterator, Sentinel>>::type
    make_iterator_range(Iterator iterator, Sentinel sentinel)
    {
        return iterator_range<Iterator, Sentinel>(iterator, sentinel);
    }
}    // namespace pika::util

namespace pika::ranges {
    template <typename I, typename S = I>
    using subrange_t = pika::util::iterator_range<I, S>;
}    // namespace pika::ranges
