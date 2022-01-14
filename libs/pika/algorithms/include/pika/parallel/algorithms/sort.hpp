//  Copyright (c) 2015 John Biddiscombe
//  Copyright (c) 2015-2020 Hartmut Kaiser
//  Copyright (c) 2015-2019 Francisco Jose Tapia
//  Copyright (c) 2018 Taeguk Kwon
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
    /// order of equal elements is not guaranteed to be preserved. The function
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
    /// The assignments in the parallel \a sort algorithm invoked without
    /// an execution policy object execute in sequential order in the
    /// calling thread.
    ///
    /// \returns  The \a sort algorithm does not return anything.
    ///
    template <typename RandomIt>
    void sort(RandomIt first, RandomIt last);

    ///////////////////////////////////////////////////////////////////////////
    /// Sorts the elements in the range [first, last) in ascending order. The
    /// order of equal elements is not guaranteed to be preserved. The function
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
    /// \returns  The \a sort algorithm returns a
    ///           \a pika::future<void> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns nothing
    ///           otherwise.
    ///
    template <typename ExPolicy, typename RandomIt>
    typename util::detail::algorithm_result<ExPolicy>::type
    sort(ExPolicy&& policy, RandomIt first, RandomIt last);

    ///////////////////////////////////////////////////////////////////////////
    /// Sorts the elements in the range [first, last) in ascending order. The
    /// order of equal elements is not guaranteed to be preserved. The function
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
    /// The assignments in the parallel \a sort algorithm invoked without
    /// an execution policy object execute in sequential order in the
    /// calling thread.
    ///
    /// \returns  The \a sort algorithm returns nothing.
    ///
    template <typename RandomIt, typename Comp, typename Proj>
    void sort(RandomIt first, RandomIt last, Comp&& comp, Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Sorts the elements in the range [first, last) in ascending order. The
    /// order of equal elements is not guaranteed to be preserved. The function
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
    /// \returns  The \a sort algorithm returns a
    ///           \a pika::future<void> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns nothing
    ///           otherwise.
    ///
    template <typename ExPolicy, typename RandomIt, typename Comp,
        typename Proj>
    typename parallel::util::detail::algorithm_result<ExPolicy>::type
    sort(ExPolicy&& policy, RandomIt first, RandomIt last, Comp&& comp,
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
#include <pika/parallel/algorithms/detail/advance_to_sentinel.hpp>
#include <pika/parallel/algorithms/detail/dispatch.hpp>
#include <pika/parallel/algorithms/detail/is_sorted.hpp>
#include <pika/parallel/algorithms/detail/pivot.hpp>
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
    // sort
    namespace detail {

        /// \cond NOINTERNAL
        static const std::size_t sort_limit_per_task = 65536ul;

        /// \brief this function is the work assigned to each thread in the
        ///        parallel process
        /// \exception
        /// \return
        /// \remarks
        template <typename ExPolicy, typename RandomIt, typename Comp>
        pika::future<RandomIt> sort_thread(ExPolicy&& policy, RandomIt first,
            RandomIt last, Comp comp, std::size_t chunk_size)
        {
            std::ptrdiff_t N = last - first;
            if (std::size_t(N) <= chunk_size)
            {
                return execution::async_execute(policy.executor(),
                    [first, last, comp = PIKA_MOVE(comp)]() -> RandomIt {
                        std::sort(first, last, comp);
                        return last;
                    });
            }

            // check if sorted
            if (detail::is_sorted_sequential(first, last, comp))
            {
                return pika::make_ready_future(last);
            }

            // pivot selections
            pivot9(first, last, comp);

            using reference =
                typename std::iterator_traits<RandomIt>::reference;

            reference val = *first;
            RandomIt c_first = first + 1, c_last = last - 1;

            while (comp(*c_first, val))
            {
                ++c_first;
            }
            while (comp(val, *c_last))
            {
                --c_last;
            }
            while (c_first < c_last)
            {
#if defined(PIKA_HAVE_CXX20_STD_RANGES_ITER_SWAP)
                std::ranges::iter_swap(c_first++, c_last--);
#else
                std::iter_swap(c_first++, c_last--);
#endif
                while (comp(*c_first, val))
                {
                    ++c_first;
                }
                while (comp(val, *c_last))
                {
                    --c_last;
                }
            }

#if defined(PIKA_HAVE_CXX20_STD_RANGES_ITER_SWAP)
            std::ranges::iter_swap(first, c_last);
#else
            std::iter_swap(first, c_last);
#endif

            // spawn tasks for each sub section
            pika::future<RandomIt> left = execution::async_execute(
                policy.executor(), &sort_thread<ExPolicy, RandomIt, Comp>,
                policy, first, c_last, comp, chunk_size);

            pika::future<RandomIt> right = execution::async_execute(
                policy.executor(), &sort_thread<ExPolicy, RandomIt, Comp>,
                policy, c_first, last, comp, chunk_size);

            return pika::dataflow(
                [last](pika::future<RandomIt>&& left,
                    pika::future<RandomIt>&& right) -> RandomIt {
                    if (left.has_exception() || right.has_exception())
                    {
                        std::list<std::exception_ptr> errors;
                        if (left.has_exception())
                            errors.push_back(left.get_exception_ptr());
                        if (right.has_exception())
                            errors.push_back(right.get_exception_ptr());

                        throw exception_list(PIKA_MOVE(errors));
                    }
                    return last;
                },
                PIKA_MOVE(left), PIKA_MOVE(right));
        }

        /// \param [in] first   iterator to the first element to sort
        /// \param [in] last    iterator to the next element after the last
        /// \param [in] comp    object for to Comp
        /// \exception
        /// \return
        /// \remarks
        template <typename ExPolicy, typename RandomIt, typename Comp>
        pika::future<RandomIt> parallel_sort_async(
            ExPolicy&& policy, RandomIt first, RandomIt last, Comp&& comp)
        {
            // number of elements to sort
            std::size_t count = last - first;

            // figure out the chunk size to use
            std::size_t const cores = execution::processing_units_count(
                policy.parameters(), policy.executor());

            std::size_t max_chunks = execution::maximal_number_of_chunks(
                policy.parameters(), policy.executor(), cores, count);

            std::size_t chunk_size = execution::get_chunk_size(
                policy.parameters(), policy.executor(),
                [](std::size_t) { return 0; }, cores, count);

            util::detail::adjust_chunk_size_and_max_chunks(
                cores, count, max_chunks, chunk_size);

            // we should not get smaller than our sort_limit_per_task
            chunk_size = (std::max)(chunk_size, sort_limit_per_task);

            std::ptrdiff_t N = last - first;
            PIKA_ASSERT(N >= 0);

            if (std::size_t(N) < chunk_size)
            {
                std::sort(first, last, comp);
                return pika::make_ready_future(last);
            }

            // check if already sorted
            if (detail::is_sorted_sequential(first, last, comp))
            {
                return pika::make_ready_future(last);
            }

            return execution::async_execute(policy.executor(),
                &sort_thread<typename std::decay<ExPolicy>::type, RandomIt,
                    Comp>,
                PIKA_FORWARD(ExPolicy, policy), first, last,
                PIKA_FORWARD(Comp, comp), chunk_size);
        }

        ///////////////////////////////////////////////////////////////////////
        // sort
        template <typename RandomIt>
        struct sort : public detail::algorithm<sort<RandomIt>, RandomIt>
        {
            sort()
              : sort::algorithm("sort")
            {
            }

            template <typename ExPolicy, typename Sent, typename Comp,
                typename Proj>
            static RandomIt sequential(
                ExPolicy, RandomIt first, Sent last, Comp&& comp, Proj&& proj)
            {
                auto last_iter = detail::advance_to_sentinel(first, last);
                std::sort(first, last_iter,
                    util::compare_projected<Comp&, Proj&>(comp, proj));
                return last_iter;
            }

            template <typename ExPolicy, typename Sent, typename Comp,
                typename Proj>
            static typename util::detail::algorithm_result<ExPolicy,
                RandomIt>::type
            parallel(ExPolicy&& policy, RandomIt first, Sent last_s,
                Comp&& comp, Proj&& proj)
            {
                auto last = detail::advance_to_sentinel(first, last_s);
                typedef util::detail::algorithm_result<ExPolicy, RandomIt>
                    algorithm_result;

                try
                {
                    // call the sort routine and return the right type,
                    // depending on execution policy
                    return algorithm_result::get(parallel_sort_async(
                        PIKA_FORWARD(ExPolicy, policy), first, last,
                        util::compare_projected<Comp&, Proj&>(comp, proj)));
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
    template <typename ExPolicy, typename RandomIt,
        typename Comp = detail::less,
        typename Proj = util::projection_identity,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy<ExPolicy>::value &&
            pika::traits::is_iterator_v<RandomIt> &&
            traits::is_projected<Proj, RandomIt>::value &&
            traits::is_indirect_callable<ExPolicy, Comp,
                traits::projected<Proj, RandomIt>,
                traits::projected<Proj, RandomIt>
            >::value
        )>
    // clang-format on
    PIKA_DEPRECATED_V(
        0, 1, "pika::parallel::sort is deprecated, use pika::sort instead")
        typename util::detail::algorithm_result<ExPolicy, RandomIt>::type
        sort(ExPolicy&& policy, RandomIt first, RandomIt last,
            Comp&& comp = Comp(), Proj&& proj = Proj())
    {
        static_assert((pika::traits::is_random_access_iterator_v<RandomIt>),
            "Requires a random access iterator.");

#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return detail::sort<RandomIt>().call(PIKA_FORWARD(ExPolicy, policy),
            first, last, PIKA_FORWARD(Comp, comp), PIKA_FORWARD(Proj, proj));
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }
}}}    // namespace pika::parallel::v1

namespace pika {
    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::sort
    inline constexpr struct sort_t final
      : pika::detail::tag_parallel_algorithm<sort_t>
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
        friend void tag_fallback_invoke(pika::sort_t, RandomIt first,
            RandomIt last, Comp&& comp = Comp(), Proj&& proj = Proj())
        {
            static_assert(pika::traits::is_random_access_iterator_v<RandomIt>,
                "Requires a random access iterator.");

            pika::parallel::v1::detail::sort<RandomIt>().call(
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
        tag_fallback_invoke(pika::sort_t, ExPolicy&& policy, RandomIt first,
            RandomIt last, Comp&& comp = Comp(), Proj&& proj = Proj())
        {
            static_assert(pika::traits::is_random_access_iterator_v<RandomIt>,
                "Requires a random access iterator.");

            using result_type =
                typename pika::parallel::util::detail::algorithm_result<
                    ExPolicy>::type;

            return pika::util::void_guard<result_type>(),
                   pika::parallel::v1::detail::sort<RandomIt>().call(
                       PIKA_FORWARD(ExPolicy, policy), first, last,
                       PIKA_FORWARD(Comp, comp), PIKA_FORWARD(Proj, proj));
        }
    } sort{};
}    // namespace pika

#endif    // DOXYGEN
