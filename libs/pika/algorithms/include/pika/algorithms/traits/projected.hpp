//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/execution/traits/is_execution_policy.hpp>
#include <pika/execution/traits/vector_pack_load_store.hpp>
#include <pika/execution/traits/vector_pack_type.hpp>
#include <pika/functional/invoke_result.hpp>
#include <pika/functional/traits/is_invocable.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/type_support/pack.hpp>

#include <iterator>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace pika::detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct projected_iterator
    {
        using type = std::decay_t<T>;
    };

    template <typename Iterator>
    struct projected_iterator<Iterator,
        std::void_t<typename std::decay_t<Iterator>::proxy_type>>
    {
        using type = typename std::decay_t<Iterator>::proxy_type;
    };
}    // namespace pika::detail

namespace pika::parallel::detail {
    template <typename F, typename Iter, typename Enable = void>
    struct projected_result_of_impl;

    template <typename Proj, typename Iter>
    struct projected_result_of_impl<Proj, Iter,
        typename std::enable_if<pika::traits::is_iterator<Iter>::value>::type>
      : pika::util::detail::invoke_result<Proj,
            typename std::iterator_traits<Iter>::reference>
    {
    };

    template <typename Projected>
    struct projected_result_of_indirect
      : projected_result_of_impl<typename Projected::projector_type,
            typename Projected::iterator_type>
    {
    };

#if defined(PIKA_HAVE_DATAPAR)
    // This is being instantiated if a vector pack execution policy is used
    // with a zip_iterator. In this case the function object is invoked
    // with a tuple<datapar<T>...> instead of just a tuple<T...>
    template <typename Proj, typename ValueType, typename Enable = void>
    struct projected_result_of_vector_pack_
      : pika::util::detail::invoke_result<Proj,
            typename pika::parallel::traits::vector_pack_load<
                typename pika::parallel::traits::vector_pack_type<
                    ValueType>::type,
                ValueType>::value_type&>
    {
    };

    template <typename Projected, typename Enable = void>
    struct projected_result_of_vector_pack;

    template <typename Projected>
    struct projected_result_of_vector_pack<Projected,
        std::void_t<typename Projected::iterator_type>>
      : projected_result_of_vector_pack_<typename Projected::projector_type,
            typename std::iterator_traits<
                typename Projected::iterator_type>::value_type>
    {
    };
#endif

    template <typename F, typename Iter, typename Enable = void>
    struct projected_result_of
      : projected_result_of_impl<std::decay_t<F>, std::decay_t<Iter>>
    {
    };

    template <typename F, typename Iter, typename Enable = void>
    struct is_projected_impl : std::false_type
    {
    };

    // the given projection function is valid, if it can be invoked using
    // the dereferenced iterator type and if the projection does not return
    // void
    template <typename Proj, typename Iter>
    struct is_projected_impl<Proj, Iter,
        typename std::enable_if<pika::traits::is_iterator<Iter>::value &&
            pika::detail::is_invocable<Proj,
                typename std::iterator_traits<Iter>::reference>::value>::type>
      : std::integral_constant<bool,
            !std::is_void<typename pika::util::detail::invoke_result<Proj,
                typename std::iterator_traits<Iter>::reference>::type>::value>
    {
    };

    template <typename Projected, typename Enable = void>
    struct is_projected_indirect_impl : std::false_type
    {
    };

    template <typename Projected>
    struct is_projected_indirect_impl<Projected,
        std::void_t<typename Projected::projector_type>>
      : is_projected_impl<typename Projected::projector_type,
            typename Projected::iterator_type>
    {
    };

    template <typename F, typename Iter, typename Enable = void>
    struct is_projected
      : is_projected_impl<std::decay_t<F>,
            typename pika::detail::projected_iterator<Iter>::type>
    {
    };

    template <typename F, typename Iter>
    using is_projected_t = typename is_projected<F, Iter>::type;

    template <typename F, typename Iter>
    inline constexpr bool is_projected_v = is_projected<F, Iter>::value;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Proj, typename Iter>
    struct projected
    {
        using projector_type = std::decay_t<Proj>;
        using iterator_type =
            typename pika::detail::projected_iterator<Iter>::type;
    };

    template <typename Projected, typename Enable = void>
    struct is_projected_indirect : is_projected_indirect_impl<Projected>
    {
    };

    template <typename Projected, typename Enable = void>
    struct is_projected_zip_iterator : std::false_type
    {
    };

    template <typename Projected>
    struct is_projected_zip_iterator<Projected,
        std::void_t<typename Projected::iterator_type>>
      : pika::traits::is_zip_iterator<typename Projected::iterator_type>
    {
    };

    template <typename F, typename... Args>
    struct is_indirect_callable_impl_base
      : pika::detail::is_invocable<F, Args...>
    {
    };

    template <typename ExPolicy, typename F, typename ProjectedPack,
        typename Enable = void>
    struct is_indirect_callable_impl : std::false_type
    {
    };

    template <typename ExPolicy, typename F, typename... Projected>
    struct is_indirect_callable_impl<ExPolicy, F,
        pika::util::detail::pack<Projected...>,
        typename std::enable_if<
            pika::util::detail::all_of<
                is_projected_indirect<Projected>...>::value &&
            (!pika::is_vectorpack_execution_policy<ExPolicy>::value ||
                !pika::util::detail::all_of<
                    is_projected_zip_iterator<Projected>...>::value)>::type>
      : is_indirect_callable_impl_base<F,
            typename projected_result_of_indirect<Projected>::type...>
    {
    };

#if defined(PIKA_HAVE_DATAPAR)
    // Vector pack execution policies used with zip-iterators require
    // special handling because zip_iterator<>::reference is not a real
    // reference type.
    template <typename ExPolicy, typename F, typename... Projected>
    struct is_indirect_callable_impl<ExPolicy, F,
        pika::util::detail::pack<Projected...>,
        typename std::enable_if<
            pika::util::detail::all_of<
                is_projected_indirect<Projected>...>::value &&
            pika::is_vectorpack_execution_policy<ExPolicy>::value &&
            pika::util::detail::all_of<
                is_projected_zip_iterator<Projected>...>::value>::type>
      : is_indirect_callable_impl_base<F,
            typename projected_result_of_vector_pack<Projected>::type...>
    {
    };
#endif

    template <typename ExPolicy, typename F, typename... Projected>
    struct is_indirect_callable
      : detail::is_indirect_callable_impl<std::decay_t<ExPolicy>,
            std::decay_t<F>,
            pika::util::detail::pack<std::decay_t<Projected>...>>
    {
    };

    template <typename ExPolicy, typename F, typename... Projected>
    using is_indirect_callable_t =
        typename is_indirect_callable<ExPolicy, F, Projected...>::type;

    template <typename ExPolicy, typename F, typename... Projected>
    inline constexpr bool is_indirect_callable_v =
        is_indirect_callable<ExPolicy, F, Projected...>::value;
}    // namespace pika::parallel::detail
