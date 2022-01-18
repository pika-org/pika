//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

#if defined(PIKA_HAVE_DATAPAR)
#include <pika/execution/traits/is_execution_policy.hpp>
#include <pika/execution/traits/vector_pack_alignment_size.hpp>
#include <pika/execution/traits/vector_pack_type.hpp>
#include <pika/functional/tag_invoke.hpp>
#include <pika/parallel/algorithms/detail/generate.hpp>
#include <pika/parallel/datapar/iterator_helpers.hpp>
#include <pika/parallel/datapar/loop.hpp>
#include <pika/parallel/util/result_types.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace pika { namespace parallel { inline namespace v1 { namespace detail {

    template <typename Iterator>
    struct datapar_generate_helper
    {
        using iterator_type = std::decay_t<Iterator>;
        using value_type =
            typename std::iterator_traits<iterator_type>::value_type;
        using V =
            typename pika::parallel::traits::vector_pack_type<value_type>::type;

        static constexpr std::size_t size = traits::vector_pack_size<V>::value;

        template <typename Iter, typename F>
        PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if_t<
            pika::parallel::util::detail::iterator_datapar_compatible<
                Iter>::value,
            Iter>
        call(Iter first, std::size_t count, F&& f)
        {
            std::size_t len = count;
            for (; !pika::parallel::util::detail::is_data_aligned(first) &&
                 len != 0;
                 --len)
            {
                *first++ = f.template operator()<value_type>();
            }

            for (std::int64_t len_v = std::int64_t(len - (size + 1)); len_v > 0;
                 len_v -= size, len -= size)
            {
                auto tmp = f.template operator()<V>();
                traits::vector_pack_store<V, value_type>::aligned(tmp, first);
                std::advance(first, size);
            }

            for (/* */; len != 0; --len)
            {
                *first++ = f.template operator()<value_type>();
            }
            return first;
        }

        template <typename Iter, typename F>
        PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if_t<
            !pika::parallel::util::detail::iterator_datapar_compatible<
                Iter>::value,
            Iter>
        call(Iter first, std::size_t count, F&& f)
        {
            while (count--)
            {
                *first++ = f.template operator()<value_type>();
            }
            return first;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    struct datapar_generate
    {
        template <typename ExPolicy, typename Iter, typename Sent, typename F>
        PIKA_HOST_DEVICE PIKA_FORCEINLINE static Iter call(
            ExPolicy&&, Iter first, Sent last, F&& f)
        {
            std::size_t count = std::distance(first, last);
            return datapar_generate_helper<Iter>::call(
                first, count, PIKA_FORWARD(F, f));
        }
    };

    template <typename ExPolicy, typename Iter, typename Sent, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE typename std::enable_if<
        pika::is_vectorpack_execution_policy<ExPolicy>::value, Iter>::type
    tag_invoke(
        sequential_generate_t, ExPolicy&& policy, Iter first, Sent last, F&& f)
    {
        return datapar_generate::call(
            PIKA_FORWARD(ExPolicy, policy), first, last, PIKA_FORWARD(F, f));
    }

    ///////////////////////////////////////////////////////////////////////////
    struct datapar_generate_n
    {
        template <typename ExPolicy, typename Iter, typename F>
        PIKA_HOST_DEVICE PIKA_FORCEINLINE static Iter call(
            ExPolicy&&, Iter first, std::size_t count, F&& f)
        {
            return datapar_generate_helper<Iter>::call(
                first, count, PIKA_FORWARD(F, f));
        }
    };

    template <typename ExPolicy, typename Iter, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE typename std::enable_if<
        pika::is_vectorpack_execution_policy<ExPolicy>::value, Iter>::type
    tag_invoke(sequential_generate_n_t, ExPolicy&& policy, Iter first,
        std::size_t count, F&& f)
    {
        return datapar_generate_n::call(
            PIKA_FORWARD(ExPolicy, policy), first, count, PIKA_FORWARD(F, f));
    }
}}}}    // namespace pika::parallel::v1::detail
#endif
