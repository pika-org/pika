//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/starts_with.hpp

#pragma once

#include <pika/local/config.hpp>
#include <pika/algorithms/traits/projected.hpp>
#include <pika/execution/algorithms/detail/predicates.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/parallel/algorithms/detail/dispatch.hpp>
#include <pika/parallel/algorithms/detail/distance.hpp>
#include <pika/parallel/algorithms/equal.hpp>
#include <pika/parallel/algorithms/mismatch.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/invoke_projected.hpp>
#include <pika/parallel/util/projection_identity.hpp>
#include <pika/parallel/util/result_types.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // starts_with
    namespace detail {
        template <typename FwdIter1, typename FwdIter2, typename Sent2>
        bool get_starts_with_result(
            util::in_in_result<FwdIter1, FwdIter2>&& p, Sent2 last2)
        {
            return p.in2 == last2;
        }

        template <typename FwdIter1, typename FwdIter2, typename Sent2>
        pika::future<bool> get_starts_with_result(
            pika::future<util::in_in_result<FwdIter1, FwdIter2>>&& f,
            Sent2 last2)
        {
            return pika::make_future<bool>(PIKA_MOVE(f),
                [last2 = PIKA_MOVE(last2)](
                    util::in_in_result<FwdIter1, FwdIter2>&& p) -> bool {
                    return p.in2 == last2;
                });
        }

        struct starts_with : public detail::algorithm<starts_with, bool>
        {
            starts_with()
              : starts_with::algorithm("starts_with")
            {
            }

            template <typename ExPolicy, typename Iter1, typename Sent1,
                typename Iter2, typename Sent2, typename Pred, typename Proj1,
                typename Proj2>
            static bool sequential(ExPolicy, Iter1 first1, Sent1 last1,
                Iter2 first2, Sent2 last2, Pred&& pred, Proj1&& proj1,
                Proj2&& proj2)
            {
                auto dist1 = detail::distance(first1, last1);
                auto dist2 = detail::distance(first2, last2);

                if (dist1 < dist2)
                {
                    return false;
                }

                auto end_first = std::next(first1, dist2);
                return detail::get_starts_with_result<Iter1, Iter2, Sent2>(
                    pika::parallel::v1::detail::mismatch_binary<
                        util::in_in_result<Iter1, Iter2>>()
                        .call(pika::execution::seq, PIKA_MOVE(first1),
                            PIKA_MOVE(end_first), PIKA_MOVE(first2), last2,
                            PIKA_FORWARD(Pred, pred), PIKA_FORWARD(Proj1, proj1),
                            PIKA_FORWARD(Proj2, proj2)),
                    last2);
            }

            template <typename ExPolicy, typename FwdIter1, typename Sent1,
                typename FwdIter2, typename Sent2, typename Pred,
                typename Proj1, typename Proj2>
            static typename util::detail::algorithm_result<ExPolicy, bool>::type
            parallel(ExPolicy&& policy, FwdIter1 first1, Sent1 last1,
                FwdIter2 first2, Sent2 last2, Pred&& pred, Proj1&& proj1,
                Proj2&& proj2)
            {
                auto dist1 = detail::distance(first1, last1);
                auto dist2 = detail::distance(first2, last2);

                if (dist1 < dist2)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        false);
                }

                auto end_first = std::next(first1, dist2);
                return detail::get_starts_with_result<FwdIter1, FwdIter2,
                    Sent2>(
                    detail::mismatch_binary<
                        util::in_in_result<FwdIter1, FwdIter2>>()
                        .call(PIKA_FORWARD(ExPolicy, policy), first1, end_first,
                            first2, last2, PIKA_FORWARD(Pred, pred),
                            PIKA_FORWARD(Proj1, proj1),
                            PIKA_FORWARD(Proj2, proj2)),
                    last2);
            }
        };
        /// \endcond
    }    // namespace detail
}}}      // namespace pika::parallel::v1

namespace pika {

    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::starts_with
    inline constexpr struct starts_with_t final
      : pika::functional::detail::tag_fallback<starts_with_t>
    {
    private:
        // clang-format off
        template <typename InIter1, typename InIter2,
            typename Pred = pika::parallel::v1::detail::equal_to,
            typename Proj1 = parallel::util::projection_identity,
            typename Proj2 = parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator<InIter1>::value &&
                pika::traits::is_iterator<InIter2>::value &&
                pika::parallel::traits::is_indirect_callable<
                    pika::execution::sequenced_policy, Pred,
                    pika::parallel::traits::projected<Proj1, InIter1>,
                    pika::parallel::traits::projected<Proj2, InIter2>
                >::value
            )>
        // clang-format on
        friend bool tag_fallback_invoke(pika::starts_with_t, InIter1 first1,
            InIter1 last1, InIter2 first2, InIter2 last2, Pred&& pred = Pred(),
            Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
        {
            static_assert(pika::traits::is_input_iterator<InIter1>::value,
                "Required at least input iterator.");

            static_assert(pika::traits::is_input_iterator<InIter2>::value,
                "Required at least input iterator.");

            return pika::parallel::v1::detail::starts_with().call(
                pika::execution::seq, first1, last1, first2, last2,
                PIKA_FORWARD(Pred, pred), PIKA_FORWARD(Proj1, proj1),
                PIKA_FORWARD(Proj2, proj2));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Pred = ranges::equal_to,
            typename Proj1 = parallel::util::projection_identity,
            typename Proj2 = parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator<FwdIter1>::value &&
                pika::traits::is_iterator<FwdIter2>::value &&
                pika::parallel::traits::is_indirect_callable<
                    ExPolicy, Pred,
                    pika::parallel::traits::projected<Proj1, FwdIter1>,
                    pika::parallel::traits::projected<Proj2, FwdIter2>
                >::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(pika::starts_with_t, ExPolicy&& policy,
            FwdIter1 first1, FwdIter1 last1, FwdIter2 first2, FwdIter2 last2,
            Pred&& pred = Pred(), Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
        {
            static_assert(pika::traits::is_forward_iterator<FwdIter1>::value,
                "Required at least forward iterator.");

            static_assert(pika::traits::is_forward_iterator<FwdIter2>::value,
                "Required at least forward iterator.");

            return pika::parallel::v1::detail::starts_with().call(
                PIKA_FORWARD(ExPolicy, policy), first1, last1, first2, last2,
                PIKA_FORWARD(Pred, pred), PIKA_FORWARD(Proj1, proj1),
                PIKA_FORWARD(Proj2, proj2));
        }
    } starts_with{};

}    // namespace pika
