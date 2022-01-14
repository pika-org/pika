//  Copyright (c) 2015-2017 Francisco Jose Tapia
//  Copyright (c) 2020 Hartmut Kaiser
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if defined(DOXYGEN)

namespace pika {
    // clang-format off

    ///////////////////////////////////////////////////////////////////////////
    /// Sorts the elements in the range [first, last) in ascending order. The
    /// relative order of equal elements is preserved. The function
    /// uses the given comparison function object comp (defaults to using
    /// operator<()).
    ///
    /// \note   Complexity: O(Nlog(N)), where N = std::distance(first, last)
    ///                     comparisons.
    ///
    /// A sequence is sorted with respect to a comparator \a comp and a
    /// projection \a proj if for every iterator i pointing to the sequence and
    /// every non-negative integer n such that i + n is a valid iterator
    /// pointing to an element of the sequence, and
    /// INVOKE(comp, INVOKE(proj, *(i + n)), INVOKE(proj, *i)) == false.
    ///
    /// \tparam RandomIt    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    ///
    /// The assignments in the parallel \a stable_sort algorithm invoked without
    /// an execution policy object execute in sequential order in the
    /// calling thread.
    ///
    /// \returns  The \a stable_sort algorithm does not return anything.
    ///
    template <typename RandomIt>
    void stable_sort(RandomIt first, RandomIt last);

    ///////////////////////////////////////////////////////////////////////////
    /// Sorts the elements in the range [first, last) in ascending order. The
    /// relative order of equal elements is preserved. The function
    /// uses the given comparison function object comp (defaults to using
    /// operator<()).
    ///
    /// \note   Complexity: O(Nlog(N)), where N = std::distance(first, last)
    ///                     comparisons.
    ///
    /// A sequence is sorted with respect to a comparator \a comp and a
    /// projection \a proj if for every iterator i pointing to the sequence and
    /// every non-negative integer n such that i + n is a valid iterator
    /// pointing to an element of the sequence, and
    /// INVOKE(comp, INVOKE(proj, *(i + n)), INVOKE(proj, *i)) == false.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam RandomIt    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
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
    /// \returns  The \a stable_sort algorithm returns a
    ///           \a pika::future<void> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns nothing
    ///           otherwise.
    ///
    template <typename ExPolicy, typename RandomIt>
    typename util::detail::algorithm_result<ExPolicy>::type
    stable_sort(ExPolicy&& policy, RandomIt first, RandomIt last);

    ///////////////////////////////////////////////////////////////////////////
    /// Sorts the elements in the range [first, last) in ascending order. The
    /// relative order of equal elements is preserved. The function
    /// uses the given comparison function object comp (defaults to using
    /// operator<()).
    ///
    /// \note   Complexity: O(Nlog(N)), where N = std::distance(first, last)
    ///                     comparisons.
    ///
    /// A sequence is sorted with respect to a comparator \a comp and a
    /// projection \a proj if for every iterator i pointing to the sequence and
    /// every non-negative integer n such that i + n is a valid iterator
    /// pointing to an element of the sequence, and
    /// INVOKE(comp, INVOKE(proj, *(i + n)), INVOKE(proj, *i)) == false.
    ///
    /// \tparam RandomIt    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param comp         comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that comp
    ///                     will not apply any non-constant function through the
    ///                     dereferenced iterator.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each pair of elements as a
    ///                     projection operation before the actual predicate
    ///                     \a comp is invoked.
    ///
    /// \a comp has to induce a strict weak ordering on the values.
    ///
    /// The assignments in the parallel \a stable_sort algorithm invoked without
    /// an execution policy object execute in sequential order in the
    /// calling thread.
    ///
    /// \returns  The \a stable_sort algorithm returns nothing.
    ///
    template <typename RandomIt, typename Comp, typename Proj>
    void stable_sort(RandomIt first, RandomIt last, Comp&& comp, Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Sorts the elements in the range [first, last) in ascending order. The
    /// relative order of equal elements is preserved. The function
    /// uses the given comparison function object comp (defaults to using
    /// operator<()).
    ///
    /// \note   Complexity: O(Nlog(N)), where N = std::distance(first, last)
    ///                     comparisons.
    ///
    /// A sequence is sorted with respect to a comparator \a comp and a
    /// projection \a proj if for every iterator i pointing to the sequence and
    /// every non-negative integer n such that i + n is a valid iterator
    /// pointing to an element of the sequence, and
    /// INVOKE(comp, INVOKE(proj, *(i + n)), INVOKE(proj, *i)) == false.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam RandomIt    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param comp         comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that comp
    ///                     will not apply any non-constant function through the
    ///                     dereferenced iterator.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each pair of elements as a
    ///                     projection operation before the actual predicate
    ///                     \a comp is invoked.
    ///
    /// \a comp has to induce a strict weak ordering on the values.
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
    /// \returns  The \a stable_sort algorithm returns a
    ///           \a pika::future<void> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns nothing
    ///           otherwise.
    ///
    template <typename ExPolicy, typename RandomIt, typename Comp,
        typename Proj>
    typename parallel::util::detail::algorithm_result<ExPolicy>::type
    stable_sort(ExPolicy&& policy, RandomIt first, RandomIt last, Comp&& comp,
        Proj&& proj);

    // clang-format on
}    // namespace pika

#else    // DOXYGEN

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/async_local/dataflow.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>

#include <pika/algorithms/traits/projected.hpp>
#include <pika/execution/algorithms/detail/predicates.hpp>
#include <pika/execution/executors/execution.hpp>
#include <pika/execution/executors/execution_information.hpp>
#include <pika/execution/executors/execution_parameters.hpp>
#include <pika/executors/exception_list.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/parallel/algorithms/detail/advance_and_get_distance.hpp>
#include <pika/parallel/algorithms/detail/advance_to_sentinel.hpp>
#include <pika/parallel/algorithms/detail/dispatch.hpp>
#include <pika/parallel/algorithms/detail/distance.hpp>
#include <pika/parallel/algorithms/detail/parallel_stable_sort.hpp>
#include <pika/parallel/algorithms/detail/spin_sort.hpp>
#include <pika/parallel/util/compare_projected.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/detail/chunk_size.hpp>
#include <pika/parallel/util/detail/sender_util.hpp>
#include <pika/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <functional>
#include <iterator>
#include <list>
#include <type_traits>
#include <utility>

namespace pika { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // stable_sort
    namespace detail {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        // stable_sort
        template <typename RandomIt>
        struct stable_sort
          : public detail::algorithm<stable_sort<RandomIt>, RandomIt>
        {
            stable_sort()
              : stable_sort::algorithm("stable_sort")
            {
            }

            template <typename ExPolicy, typename Sentinel, typename Compare,
                typename Proj>
            static RandomIt sequential(ExPolicy, RandomIt first, Sentinel last,
                Compare&& comp, Proj&& proj)
            {
                using compare_type = util::compare_projected<Compare&, Proj&>;

                auto last_iter = detail::advance_to_sentinel(first, last);

                spin_sort(first, last_iter, compare_type(comp, proj));
                return last_iter;
            }

            template <typename ExPolicy, typename Sentinel, typename Compare,
                typename Proj>
            static typename util::detail::algorithm_result<ExPolicy,
                RandomIt>::type
            parallel(ExPolicy&& policy, RandomIt first, Sentinel last,
                Compare&& compare, Proj&& proj)
            {
                using algorithm_result =
                    util::detail::algorithm_result<ExPolicy, RandomIt>;
                using compare_type = util::compare_projected<Compare&, Proj&>;

                // number of elements to sort
                auto last_iter = first;
                std::size_t count =
                    detail::advance_and_get_distance(last_iter, last);

                // figure out the chunk size to use
                std::size_t cores = execution::processing_units_count(
                    policy.parameters(), policy.executor());

                std::size_t max_chunks = execution::maximal_number_of_chunks(
                    policy.parameters(), policy.executor(), cores, count);

                std::size_t chunk_size = execution::get_chunk_size(
                    policy.parameters(), policy.executor(),
                    [](std::size_t) { return 0; }, cores, count);

                util::detail::adjust_chunk_size_and_max_chunks(
                    cores, count, max_chunks, chunk_size);

                // we should not get smaller than our sort_limit_per_task
                chunk_size = (std::max)(chunk_size, stable_sort_limit_per_task);

                try
                {
                    // call the sort routine and return the right type,
                    // depending on execution policy
                    compare_type comp(compare, proj);

                    return algorithm_result::get(
                        parallel_stable_sort(policy.executor(), first,
                            last_iter, cores, chunk_size, PIKA_MOVE(comp)));
                }
                catch (...)
                {
                    return algorithm_result::get(
                        detail::handle_exception<ExPolicy, RandomIt>::call(
                            std::current_exception()));
                }
            }
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename RandomIt, typename Sentinel,
        typename Proj = util::projection_identity,
        typename Compare = detail::less,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy<ExPolicy>::value &&
            pika::traits::is_iterator_v<RandomIt> &&
            pika::traits::is_sentinel_for<Sentinel, RandomIt>::value &&
            traits::is_projected<Proj, RandomIt>::value &&
            traits::is_indirect_callable<ExPolicy, Compare,
                traits::projected<Proj, RandomIt>,
                traits::projected<Proj, RandomIt>
            >::value
        )>
    // clang-format on
    PIKA_DEPRECATED_V(0, 1,
        "pika::parallel::stable_sort is deprecated, use pika::stable_sort "
        "instead")
        typename util::detail::algorithm_result<ExPolicy, RandomIt>::type
        stable_sort(ExPolicy&& policy, RandomIt first, Sentinel last,
            Compare&& comp = Compare(), Proj&& proj = Proj())
    {
        static_assert((pika::traits::is_random_access_iterator_v<RandomIt>),
            "Requires a random access iterator.");

#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return detail::stable_sort<RandomIt>().call(
            PIKA_FORWARD(ExPolicy, policy), first, last,
            PIKA_FORWARD(Compare, comp), PIKA_FORWARD(Proj, proj));
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }
}}}    // namespace pika::parallel::v1

namespace pika {
    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::stable_sort
    inline constexpr struct stable_sort_t final
      : pika::detail::tag_parallel_algorithm<stable_sort_t>
    {
        // clang-format off
        template <typename RandomIt,
            typename Comp = pika::parallel::v1::detail::less,
            typename Proj = parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator_v<RandomIt> &&
                parallel::traits::is_projected<Proj, RandomIt>::value &&
                parallel::traits::is_indirect_callable<
                    pika::execution::sequenced_policy, Comp,
                    parallel::traits::projected<Proj, RandomIt>,
                    parallel::traits::projected<Proj, RandomIt>
                >::value
            )>
        // clang-format on
        friend void tag_fallback_invoke(pika::stable_sort_t, RandomIt first,
            RandomIt last, Comp&& comp = Comp(), Proj&& proj = Proj())
        {
            static_assert(pika::traits::is_random_access_iterator_v<RandomIt>,
                "Requires a random access iterator.");

            pika::parallel::v1::detail::stable_sort<RandomIt>().call(
                pika::execution::seq, first, last, PIKA_FORWARD(Comp, comp),
                PIKA_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename RandomIt,
            typename Comp = pika::parallel::v1::detail::less,
            typename Proj = parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator_v<RandomIt> &&
                parallel::traits::is_projected<Proj, RandomIt>::value &&
                parallel::traits::is_indirect_callable<ExPolicy, Comp,
                    parallel::traits::projected<Proj, RandomIt>,
                    parallel::traits::projected<Proj, RandomIt>
                >::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy>::type
        tag_fallback_invoke(pika::stable_sort_t, ExPolicy&& policy,
            RandomIt first, RandomIt last, Comp&& comp = Comp(),
            Proj&& proj = Proj())
        {
            static_assert(pika::traits::is_random_access_iterator_v<RandomIt>,
                "Requires a random access iterator.");

            using result_type =
                typename pika::parallel::util::detail::algorithm_result<
                    ExPolicy>::type;

            return pika::util::void_guard<result_type>(),
                   pika::parallel::v1::detail::stable_sort<RandomIt>().call(
                       PIKA_FORWARD(ExPolicy, policy), first, last,
                       PIKA_FORWARD(Comp, comp), PIKA_FORWARD(Proj, proj));
        }
    } stable_sort{};
}    // namespace pika

#endif
