//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  This file is based on the following cppreference possible implementation:
//  https://en.cppreference.com/w/cpp/algorithm/ranges/search
#pragma once

#include <pika/local/config.hpp>
#include <pika/algorithms/traits/projected.hpp>
#include <pika/execution/algorithms/detail/predicates.hpp>
#include <pika/functional/detail/invoke.hpp>
#include <pika/parallel/algorithms/detail/dispatch.hpp>
#include <pika/parallel/algorithms/detail/distance.hpp>
#include <pika/parallel/util/compare_projected.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/loop.hpp>
#include <pika/parallel/util/partitioner.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika { namespace parallel { inline namespace v1 { namespace detail {
    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    // search
    template <typename FwdIter, typename Sent>
    struct search : public detail::algorithm<search<FwdIter, Sent>, FwdIter>
    {
        search()
          : search::algorithm("search")
        {
        }

        template <typename ExPolicy, typename FwdIter2, typename Sent2,
            typename Pred, typename Proj1, typename Proj2>
        static FwdIter sequential(ExPolicy, FwdIter first, Sent last,
            FwdIter2 s_first, Sent2 s_last, Pred&& op, Proj1&& proj1,
            Proj2&& proj2)
        {
            for (;; ++first)
            {
                FwdIter it1 = first;
                for (FwdIter2 it2 = s_first;; ++it1, ++it2)
                {
                    if (it2 == s_last)
                        return first;
                    if (it1 == last)
                        return it1;
                    if (!PIKA_INVOKE(op, PIKA_INVOKE(proj1, *it1),
                            PIKA_INVOKE(proj2, *it2)))
                        break;
                }
            }
        }

        template <typename ExPolicy, typename FwdIter2, typename Sent2,
            typename Pred, typename Proj1, typename Proj2>
        static typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        parallel(ExPolicy&& policy, FwdIter first, Sent last, FwdIter2 s_first,
            Sent2 s_last, Pred&& op, Proj1&& proj1, Proj2&& proj2)
        {
            using reference = typename std::iterator_traits<FwdIter>::reference;

            using difference_type =
                typename std::iterator_traits<FwdIter>::difference_type;

            using s_difference_type =
                typename std::iterator_traits<FwdIter2>::difference_type;

            using result =
                pika::parallel::util::detail::algorithm_result<ExPolicy,
                    FwdIter>;

            // Use of pika::distance instead of std::distance to support
            // sentinels
            s_difference_type diff =
                pika::parallel::v1::detail::distance(s_first, s_last);
            if (diff <= 0)
                return result::get(PIKA_MOVE(first));

            difference_type count =
                pika::parallel::v1::detail::distance(first, last);
            if (diff > count)
            {
                std::advance(first,
                    pika::parallel::v1::detail::distance(first, last) - 1);
                return result::get(PIKA_MOVE(first));
            }

            using partitioner =
                pika::parallel::util::partitioner<ExPolicy, FwdIter, void>;

            pika::parallel::util::cancellation_token<difference_type> tok(count);

            auto f1 = [diff, count, tok, s_first, op = PIKA_FORWARD(Pred, op),
                          proj1 = PIKA_FORWARD(Proj1, proj1),
                          proj2 = PIKA_FORWARD(Proj2, proj2)](FwdIter it,
                          std::size_t part_size,
                          std::size_t base_idx) mutable -> void {
                FwdIter curr = it;

                pika::parallel::util::loop_idx_n<std::decay_t<ExPolicy>>(
                    base_idx, it, part_size, tok,
                    [diff, count, s_first, &tok, &curr,
                        op = PIKA_FORWARD(Pred, op),
                        proj1 = PIKA_FORWARD(Proj1, proj1),
                        proj2 = PIKA_FORWARD(Proj2, proj2)](
                        reference v, std::size_t i) -> void {
                        ++curr;
                        if (PIKA_INVOKE(op, PIKA_INVOKE(proj1, v),
                                PIKA_INVOKE(proj2, *s_first)))
                        {
                            difference_type local_count = 1;
                            FwdIter2 needle = s_first;
                            FwdIter mid = curr;

                            for (difference_type len = 0;
                                 local_count != diff && len != count;
                                 ++local_count, ++len, ++mid)
                            {
                                if (!PIKA_INVOKE(op, PIKA_INVOKE(proj1, *mid),
                                        PIKA_INVOKE(proj2, *++needle)))
                                    break;
                            }

                            if (local_count == diff)
                                tok.cancel(i);
                        }
                    });
            };

            auto f2 =
                [=](std::vector<pika::future<void>>&& data) mutable -> FwdIter {
                // make sure iterators embedded in function object that is
                // attached to futures are invalidated
                data.clear();
                difference_type search_res = tok.get_data();
                if (search_res != count)
                    std::advance(first, search_res);
                else
                    std::advance(first,
                        pika::parallel::v1::detail::distance(first, last) - 1);

                return PIKA_MOVE(first);
            };
            return partitioner::call_with_index(PIKA_FORWARD(ExPolicy, policy),
                first, count - (diff - 1), 1, PIKA_MOVE(f1), PIKA_MOVE(f2));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // search_n
    template <typename FwdIter, typename Sent>
    struct search_n : public detail::algorithm<search_n<FwdIter, Sent>, FwdIter>
    {
        search_n()
          : search_n::algorithm("search_n")
        {
        }

        template <typename ExPolicy, typename FwdIter2, typename Pred,
            typename Proj1, typename Proj2>
        static FwdIter sequential(ExPolicy, FwdIter first, std::size_t count,
            FwdIter2 s_first, FwdIter2 s_last, Pred&& op, Proj1&& proj1,
            Proj2&& proj2)
        {
            return std::search(first, std::next(first, count), s_first, s_last,
                util::compare_projected<Pred&, Proj1&, Proj2&>(
                    op, proj1, proj2));
        }

        template <typename ExPolicy, typename FwdIter2, typename Pred,
            typename Proj1, typename Proj2>
        static typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        parallel(ExPolicy&& policy, FwdIter first, std::size_t count,
            FwdIter2 s_first, FwdIter2 s_last, Pred&& op, Proj1&& proj1,
            Proj2&& proj2)
        {
            typedef typename std::iterator_traits<FwdIter>::reference reference;
            typedef typename std::iterator_traits<FwdIter>::difference_type
                difference_type;
            typedef typename std::iterator_traits<FwdIter2>::difference_type
                s_difference_type;
            typedef util::detail::algorithm_result<ExPolicy, FwdIter> result;

            s_difference_type diff = std::distance(s_first, s_last);
            if (diff <= 0)
                return result::get(PIKA_MOVE(first));

            if (diff > s_difference_type(count))
                return result::get(PIKA_MOVE(first));

            typedef util::partitioner<ExPolicy, FwdIter, void> partitioner;

            util::cancellation_token<difference_type> tok(count);

            auto f1 = [count, diff, tok, s_first, op = PIKA_FORWARD(Pred, op),
                          proj1 = PIKA_FORWARD(Proj1, proj1),
                          proj2 = PIKA_FORWARD(Proj2, proj2)](FwdIter it,
                          std::size_t part_size,
                          std::size_t base_idx) mutable -> void {
                FwdIter curr = it;

                util::loop_idx_n<std::decay_t<ExPolicy>>(base_idx, it,
                    part_size, tok,
                    [count, diff, s_first, &tok, &curr,
                        op = PIKA_FORWARD(Pred, op),
                        proj1 = PIKA_FORWARD(Proj1, proj1),
                        proj2 = PIKA_FORWARD(Proj2, proj2)](
                        reference v, std::size_t i) -> void {
                        ++curr;
                        if (PIKA_INVOKE(op, PIKA_INVOKE(proj1, v),
                                PIKA_INVOKE(proj2, *s_first)))
                        {
                            difference_type local_count = 1;
                            FwdIter2 needle = s_first;
                            FwdIter mid = curr;

                            for (difference_type len = 0; local_count != diff &&
                                 len != difference_type(count);
                                 ++local_count, ++len, ++mid)
                            {
                                if (!PIKA_INVOKE(op, PIKA_INVOKE(proj1, *mid),
                                        PIKA_INVOKE(proj2, *++needle)))
                                    break;
                            }

                            if (local_count == diff)
                                tok.cancel(i);
                        }
                    });
            };

            auto f2 =
                [=](std::vector<pika::future<void>>&& data) mutable -> FwdIter {
                // make sure iterators embedded in function object that is
                // attached to futures are invalidated
                data.clear();
                difference_type search_res = tok.get_data();
                if (search_res != s_difference_type(count))
                    std::advance(first, search_res);

                return PIKA_MOVE(first);
            };
            return partitioner::call_with_index(PIKA_FORWARD(ExPolicy, policy),
                first, count - (diff - 1), 1, PIKA_MOVE(f1), PIKA_MOVE(f2));
        }
    };

    /// \endcond
}}}}    // namespace pika::parallel::v1::detail
