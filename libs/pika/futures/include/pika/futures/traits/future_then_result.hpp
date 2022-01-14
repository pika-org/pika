//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/functional/invoke_result.hpp>
#include <pika/futures/traits/future_traits.hpp>
#include <pika/futures/traits/is_future.hpp>
#include <pika/type_support/always_void.hpp>
#include <pika/type_support/identity.hpp>
#include <pika/type_support/lazy_conditional.hpp>

#include <type_traits>
#include <utility>

namespace pika { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        struct no_executor
        {
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Future, typename F>
        struct continuation_not_callable
        {
            static auto error(Future future, F& f)
            {
                f(PIKA_MOVE(future));
            }

            using type =
                decltype(error(std::declval<Future>(), std::declval<F&>()));
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Future, typename F, typename Enable = void>
        struct future_then_result
        {
            typedef typename continuation_not_callable<Future, F>::type type;
        };

        template <typename Future, typename F>
        struct future_then_result<Future, F,
            typename pika::util::always_void<
                pika::util::invoke_result_t<F&, Future>>::type>
        {
            using cont_result = pika::util::invoke_result_t<F&, Future>;

            // perform unwrapping of future<future<R>>
            using result_type = util::lazy_conditional_t<
                pika::traits::detail::is_unique_future<cont_result>::value,
                pika::traits::future_traits<cont_result>,
                pika::util::identity<cont_result>>;

            using type = pika::future<result_type>;
        };

    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename F>
    struct future_then_result : detail::future_then_result<Future, F>
    {
    };

    template <typename Future, typename F>
    using future_then_result_t = typename future_then_result<Future, F>::type;

}}    // namespace pika::traits
