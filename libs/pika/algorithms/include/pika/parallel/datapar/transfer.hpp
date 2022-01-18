//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

#if defined(PIKA_HAVE_DATAPAR)
#include <pika/execution/traits/is_execution_policy.hpp>
#include <pika/executors/datapar/execution_policy.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/functional/tag_invoke.hpp>
#include <pika/parallel/datapar/transform_loop.hpp>
#include <pika/parallel/util/result_types.hpp>
#include <pika/parallel/util/transfer.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace pika { namespace parallel { namespace util {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Iterator>
        struct datapar_copy_n
        {
            template <typename InIter, typename OutIter>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                iterators_datapar_compatible<InIter, OutIter>::value &&
                    iterator_datapar_compatible<InIter>::value &&
                    iterator_datapar_compatible<OutIter>::value,
                in_out_result<InIter, OutIter>>::type
            call(InIter first, std::size_t count, OutIter dest)
            {
                auto ret =
                    util::transform_loop_n_ind<pika::execution::simd_policy>(
                        first, count, dest, [](auto& v) { return v; });

                return util::in_out_result<InIter, OutIter>{
                    PIKA_MOVE(ret.first), PIKA_MOVE(ret.second)};
            }

            template <typename InIter, typename OutIter>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                !iterators_datapar_compatible<InIter, OutIter>::value ||
                    !iterator_datapar_compatible<InIter>::value ||
                    !iterator_datapar_compatible<OutIter>::value,
                in_out_result<InIter, OutIter>>::type
            call(InIter first, std::size_t count, OutIter dest)
            {
                return util::copy_n<pika::execution::sequenced_policy>(
                    first, count, dest);
            }
        };
    }    // namespace detail

    template <typename ExPolicy, typename InIter, typename OutIter>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE typename std::enable_if<
        pika::is_vectorpack_execution_policy<ExPolicy>::value,
        in_out_result<InIter, OutIter>>::type
    tag_invoke(pika::parallel::util::copy_n_t<ExPolicy>, InIter first,
        std::size_t count, OutIter dest)
    {
        return detail::datapar_copy_n<InIter>::call(first, count, dest);
    }
}}}    // namespace pika::parallel::util
#endif
