//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2016 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/concepts/has_member_xxx.hpp>
#include <pika/futures/traits/is_future.hpp>
#include <pika/futures/traits/is_future_range.hpp>
#include <pika/iterator_support/range.hpp>
#include <pika/iterator_support/traits/is_range.hpp>
#include <pika/util/detail/reserve.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace pika { namespace traits {
    namespace detail {
        template <typename T, typename Enable = void>
        struct acquire_future_impl;
    }

    template <typename T, typename Enable = void>
    struct acquire_future
      : detail::acquire_future_impl<typename std::decay<T>::type>
    {
    };

    template <typename T>
    using acquire_future_t = typename acquire_future<T>::type;

    struct acquire_future_disp
    {
        template <typename T>
        PIKA_FORCEINLINE acquire_future_t<T> operator()(T&& t) const
        {
            return acquire_future<T>()(PIKA_FORWARD(T, t));
        }
    };

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Enable>
        struct acquire_future_impl
        {
            static_assert(!is_future_or_future_range_v<T>,
                "!is_future_or_future_range_v<T>");

            using type = T;

            template <typename T_>
            PIKA_FORCEINLINE T operator()(T_&& value) const
            {
                return PIKA_FORWARD(T_, value);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename R>
        struct acquire_future_impl<pika::future<R>>
        {
            using type = pika::future<R>;

            PIKA_FORCEINLINE pika::future<R> operator()(
                pika::future<R>& future) const noexcept
            {
                return PIKA_MOVE(future);
            }

            PIKA_FORCEINLINE pika::future<R> operator()(
                pika::future<R>&& future) const noexcept
            {
                return PIKA_MOVE(future);
            }
        };

        template <typename R>
        struct acquire_future_impl<pika::shared_future<R>>
        {
            using type = pika::shared_future<R>;

            PIKA_FORCEINLINE pika::shared_future<R> operator()(
                pika::shared_future<R> const& future) const
            {
                return future;
            }

            PIKA_FORCEINLINE pika::shared_future<R> operator()(
                pika::shared_future<R>&& future) const noexcept
            {
                return PIKA_MOVE(future);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        PIKA_HAS_MEMBER_XXX_TRAIT_DEF(push_back)

        ///////////////////////////////////////////////////////////////////////
        template <typename Range>
        struct acquire_future_impl<Range,
            std::enable_if_t<pika::traits::is_future_range_v<Range>>>
        {
            using future_type =
                typename traits::future_range_traits<Range>::future_type;
            using type = Range;

            template <typename Range_>
            std::enable_if_t<has_push_back_v<std::decay_t<Range_>>>
            transform_future_disp(Range_&& futures, Range& values) const
            {
                detail::reserve_if_random_access_by_range(values, futures);
                std::transform(util::begin(futures), util::end(futures),
                    std::back_inserter(values), acquire_future_disp());
            }

            template <typename Range_>
            typename std::enable_if<
                !has_push_back<typename std::decay<Range_>::type>::value>::type
            transform_future_disp(Range_&& futures, Range& values) const
            {
                detail::reserve_if_random_access_by_range(values, futures);
                std::transform(util::begin(futures), util::end(futures),
                    util::begin(values), acquire_future_disp());
            }

            template <typename Range_>
            PIKA_FORCEINLINE Range operator()(Range_&& futures) const
            {
                Range values;
                transform_future_disp(PIKA_FORWARD(Range_, futures), values);
                return values;
            }
        };
    }    // namespace detail
}}       // namespace pika::traits
