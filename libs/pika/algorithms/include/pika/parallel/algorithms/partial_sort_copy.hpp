//  Copyright (c) 2020 Francisco Jose Tapia
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/partial_sort_copy.hpp

#pragma once

#if defined(DOXYGEN)

namespace pika {
    // clang-format off

    ///////////////////////////////////////////////////////////////////////////
    /// Sorts some of the elements in the range [first, last) in ascending
    /// order, storing the result in the range [d_first, d_last). At most
    /// d_last - d_first of the elements are placed sorted to the range
    /// [d_first, d_first + n) where n is the number of elements to sort
    /// (n = min(last - first, d_last - d_first)).
    ///
    /// \note   Complexity: O(Nlog(min(D,N))), where N =
    ///         std::distance(first, last) and D = std::distance(d_first,
    ///         d_last) comparisons.
    ///
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam RandIter    The type of the destination iterators used(deduced)
    ///                     This iterator type must meet the requirements of an
    ///                     random iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced). Comp defaults to detail::less.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param d_first      Refers to the beginning of the destination range.
    /// \param d_last       Refers to the end of the destination range.
    /// \param comp         comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that
    ///                     comp will not apply any non-constant function
    ///                     through the dereferenced iterator. This defaults to
    ///                     detail::less.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a partial_sort_copy algorithm returns a
    ///           returns \a RandomIt.
    ///           The algorithm returns an iterator to the element defining
    ///           the upper boundary of the sorted range i.e.
    ///           d_first + min(last - first, d_last - d_first)
    ///
    template <typename InIter, typename RandIter,
        typename Comp>
    RandIter partial_sort_copy(InIter first, InIter last, RandIter d_first,
        RandIter d_last, Comp&& comp = Comp());

    ///////////////////////////////////////////////////////////////////////////
    /// Sorts some of the elements in the range [first, last) in ascending
    /// order, storing the result in the range [d_first, d_last). At most
    /// d_last - d_first of the elements are placed sorted to the range
    /// [d_first, d_first + n) where n is the number of elements to sort
    /// (n = min(last - first, d_last - d_first)).
    ///
    /// \note   Complexity: O(Nlog(min(D,N))), where N =
    ///         std::distance(first, last) and D = std::distance(d_first,
    ///         d_last) comparisons.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam RandIter    The type of the destination iterators used(deduced)
    ///                     This iterator type must meet the requirements of an
    ///                     random iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced). Comp defaults to detail::less.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param d_first      Refers to the beginning of the destination range.
    /// \param d_last       Refers to the end of the destination range.
    /// \param comp         comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that
    ///                     comp will not apply any non-constant function
    ///                     through the dereferenced iterator. This defaults to
    ///                     detail::less.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a partial_sort_copy algorithm returns a
    ///           \a pika::future<RandomIt> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a RandomIt
    ///           otherwise.
    ///           The algorithm returns an iterator to the element defining
    ///           the upper boundary of the sorted range i.e.
    ///           d_first + min(last - first, d_last - d_first)
    ///
    template <typename ExPolicy, typename FwdIter, typename RandIter,
        typename Comp>
    parallel::util::detail::algorithm_result_t<ExPolicy, RandIter>
    partial_sort_copy(
        ExPolicy&& policy, FwdIter first, FwdIter last, RandIter d_first,
        RandIter d_last, Comp&& comp = Comp());

    // clang-format on
}    // namespace pika

#else    // DOXYGEN

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>

#include <pika/algorithms/traits/projected.hpp>
#include <pika/execution/algorithms/detail/predicates.hpp>
#include <pika/executors/exception_list.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/parallel/algorithms/copy.hpp>
#include <pika/parallel/algorithms/detail/advance_to_sentinel.hpp>
#include <pika/parallel/algorithms/detail/dispatch.hpp>
#include <pika/parallel/algorithms/detail/is_sorted.hpp>
#include <pika/parallel/algorithms/partial_sort.hpp>
#include <pika/parallel/util/compare_projected.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/detail/sender_util.hpp>
#include <pika/parallel/util/projection_identity.hpp>
#include <pika/parallel/util/result_types.hpp>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace pika { namespace parallel { inline namespace v1 { namespace detail {
    ///////////////////////////////////////////////////////////////////////
    // partial_sort_copy
    template <typename Iter>
    struct partial_sort_copy
      : public detail::algorithm<partial_sort_copy<Iter>, Iter>
    {
        partial_sort_copy()
          : partial_sort_copy::algorithm("partial_sort_copy")
        {
        }

        ///////////////////////////////////////////////////////////////////////////
        ///
        /// \brief : Sorts some of the elements in the range [first, last) in
        ///          ascending order, storing the result in the range
        ///          [d_first, d_last).
        ///          At most d_last - d_first of the elements are placed sorted to
        ///          the range [d_first, d_first + n). n is the number of elements
        ///          to sort (n = min(last - first, d_last - d_first)). The order
        ///          of equal elements is not guaranteed to be preserved.
        ///
        /// \param first : iterator to the first element
        /// \param last: iterator after the last element to be sorted
        /// \param d_first : iterator to the first element where copy the results
        /// \param d_last : iterator to the element after the end where coy the
        ///                 results
        /// \param comp : object for to compare elements
        ///
        /// \return iterator after the last element sorted
        ///
        template <typename ExPolicy, typename InIter, typename Sent1,
            typename RandIter, typename Sent2, typename Compare, typename Proj1,
            typename Proj2>
        static util::in_out_result<InIter, RandIter> sequential(ExPolicy,
            InIter first, Sent1 last, RandIter d_first, Sent2 d_last,
            Compare&& comp, Proj1&& proj1, Proj2&& proj2)
        {
            auto last_iter = detail::advance_to_sentinel(first, last);
            auto d_last_iter = detail::advance_to_sentinel(d_first, d_last);

            using value_t = typename std::iterator_traits<InIter>::value_type;
            using value1_t =
                typename std::iterator_traits<RandIter>::value_type;
            using vec_iter_t = typename std::vector<value_t>::iterator;

            static_assert(
                std::is_same_v<value1_t, value_t>, "Incompatible iterators\n");

            if ((last_iter == first) || (d_last_iter == d_first))
                return util::in_out_result<InIter, RandIter>{
                    last_iter, d_first};

            std::vector<value_t> aux(first, last_iter);
            std::int64_t noutput = d_last_iter - d_first;
            std::int64_t ninput = aux.size();

            PIKA_ASSERT(ninput >= 0 || noutput >= 0);

            util::compare_projected<Compare&, Proj1&, Proj2&> proj_comp{
                comp, proj1, proj2};

            auto nmin = ninput < noutput ? ninput : noutput;
            if (noutput >= ninput)
            {
                detail::sort<vec_iter_t>().call(pika::execution::seq,
                    aux.begin(), aux.end(), PIKA_MOVE(proj_comp),
                    util::projection_identity{});
            }
            else
            {
                parallel::v1::partial_sort<vec_iter_t>().call(
                    pika::execution::seq, aux.begin(), aux.begin() + nmin,
                    aux.end(), PIKA_MOVE(proj_comp),
                    util::projection_identity{});
            }

            detail::copy<util::in_out_result<vec_iter_t, RandIter>>().call(
                pika::execution::seq, aux.begin(), aux.begin() + nmin, d_first);
            return util::in_out_result<InIter, RandIter>{
                last_iter, d_first + nmin};
        }

        //////////////////////////////////////////////////////////////////////////
        ///
        /// \brief : Sorts some of the elements in the range [first, last) in
        ///          ascending order, storing the result in the range
        ///          [d_first, d_last).
        ///          At most d_last - d_first of the elements are placed sorted to
        ///          the range [d_first, d_first + n). n is the number of elements
        ///          to sort (n = min(last - first, d_last - d_first)). The order
        ///          of equal elements is not guaranteed to be preserved.
        ///
        /// \param first : iterator to the first element
        /// \param last: iterator after the last element to be sorted
        /// \param d_first : iterator to the firstelement where copy the results
        /// \param d_last : iterator to the element after the end where coy the
        ///                 results
        /// \param comp : object for to compare elements
        ///
        /// \return iterator after the last element sorted
        ///
        template <typename ExPolicy, typename FwdIter, typename Sent1,
            typename RandIter, typename Sent2, typename Compare, typename Proj1,
            typename Proj2>
        static util::detail::algorithm_result_t<ExPolicy,
            util::in_out_result<FwdIter, RandIter>>
        parallel(ExPolicy&& policy, FwdIter first, Sent1 last, RandIter d_first,
            Sent2 d_last, Compare&& comp, Proj1&& proj1, Proj2&& proj2)
        {
            using result_type = util::detail::algorithm_result<ExPolicy,
                util::in_out_result<FwdIter, RandIter>>;
            using value_t = typename std::iterator_traits<FwdIter>::value_type;
            using value1_t =
                typename std::iterator_traits<RandIter>::value_type;
            using vec_iter_t = typename std::vector<value_t>::iterator;

            static_assert(
                std::is_same_v<value1_t, value_t>, "Incompatible iterators\n");

            try
            {
                auto last_iter = detail::advance_to_sentinel(first, last);
                auto d_last_iter = detail::advance_to_sentinel(d_first, d_last);

                if ((last_iter == first) || (d_last_iter == d_first))
                    return result_type::get(
                        util::in_out_result<FwdIter, RandIter>{
                            last_iter, d_first});

                std::vector<value_t> aux(first, last_iter);
                std::int64_t ninput = aux.size();
                std::int64_t noutput = d_last_iter - d_first;
                PIKA_ASSERT(ninput >= 0 and noutput >= 0);

                util::compare_projected<Compare&, Proj1&, Proj2&> proj_comp{
                    comp, proj1, proj2};

                auto nmin = ninput < noutput ? ninput : noutput;
                if (noutput >= ninput)
                {
                    detail::sort<vec_iter_t>().call(
                        policy(pika::execution::non_task), aux.begin(),
                        aux.end(), PIKA_MOVE(proj_comp),
                        util::projection_identity{});
                }
                else
                {
                    //
                    pika::parallel::v1::partial_sort<vec_iter_t>().call(
                        policy(pika::execution::non_task), aux.begin(),
                        aux.begin() + nmin, aux.end(), PIKA_MOVE(proj_comp),
                        util::projection_identity{});
                };

                detail::copy<util::in_out_result<vec_iter_t, RandIter>>().call(
                    policy(pika::execution::non_task), aux.begin(),
                    aux.begin() + nmin, d_first);

                return result_type::get(util::in_out_result<FwdIter, RandIter>{
                    last_iter, d_first + nmin});
            }
            catch (...)
            {
                return result_type::get(detail::handle_exception<ExPolicy,
                    util::in_out_result<FwdIter,
                        RandIter>>::call(std::current_exception()));
            }
        }
    };
}}}}    // namespace pika::parallel::v1::detail

namespace pika {
    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::partial_sort_copy
    inline constexpr struct partial_sort_copy_t final
      : pika::detail::tag_parallel_algorithm<partial_sort_copy_t>
    {
    private:
        // clang-format off
        template <typename InIter, typename RandIter,
            typename Comp = pika::parallel::v1::detail::less,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator_v<InIter> &&
                pika::traits::is_iterator_v<RandIter> &&
                pika::is_invocable_v<Comp,
                    typename std::iterator_traits<InIter>::value_type,
                    typename std::iterator_traits<InIter>::value_type
                >
            )>
        // clang-format on
        friend RandIter tag_fallback_invoke(pika::partial_sort_copy_t,
            InIter first, InIter last, RandIter d_first, RandIter d_last,
            Comp&& comp = Comp())
        {
            static_assert(pika::traits::is_input_iterator_v<InIter>,
                "Requires at least input iterator.");

            static_assert(pika::traits::is_random_access_iterator_v<RandIter>,
                "Requires at least random access iterator.");

            using result_type = parallel::util::in_out_result<InIter, RandIter>;

            return parallel::util::get_second_element(
                parallel::v1::detail::partial_sort_copy<result_type>().call(
                    pika::execution::seq, first, last, d_first, d_last,
                    PIKA_FORWARD(Comp, comp),
                    parallel::util::projection_identity{},
                    parallel::util::projection_identity{}));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter,
            typename RandIter,
            typename Comp = pika::parallel::v1::detail::less,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy_v<ExPolicy> &&
                pika::traits::is_iterator_v<FwdIter> &&
                pika::traits::is_iterator_v<RandIter> &&
                pika::is_invocable_v<Comp,
                    typename std::iterator_traits<FwdIter>::value_type,
                    typename std::iterator_traits<FwdIter>::value_type
                >
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy, RandIter>
        tag_fallback_invoke(pika::partial_sort_copy_t, ExPolicy&& policy,
            FwdIter first, FwdIter last, RandIter d_first, RandIter d_last,
            Comp&& comp = Comp())
        {
            static_assert(pika::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            static_assert(pika::traits::is_random_access_iterator_v<RandIter>,
                "Requires at least random access iterator.");

            using result_type =
                parallel::util::in_out_result<FwdIter, RandIter>;

            return parallel::util::get_second_element(
                parallel::v1::detail::partial_sort_copy<result_type>().call(
                    PIKA_FORWARD(ExPolicy, policy), first, last, d_first, d_last,
                    PIKA_FORWARD(Comp, comp),
                    parallel::util::projection_identity{},
                    parallel::util::projection_identity{}));
        }
    } partial_sort_copy{};
}    // namespace pika

#endif    // DOXYGEN
