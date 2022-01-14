//  Copyright (c) 2015 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/make_heap.hpp

#pragma once

#if defined(DOXYGEN)
namespace pika {
    // clang-format off

    /// Constructs a \a max \a heap in the range [first, last).
    ///
    /// \note Complexity: at most (3*N) comparisons where
    ///       \a N = distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution of
    ///                     the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam RndIter     The type of the source iterators used for algorithm.
    ///                     This iterator must meet the requirements for a
    ///                     random access iterator.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of that the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     that the algorithm will be applied to.
    /// \param comp         Refers to the binary predicate which returns true
    ///                     if the first argument should be treated as less than
    ///                     the second. The signature of the function should be
    ///                     equivalent to
    ///                     \code
    ///                     bool comp(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that objects of
    ///                     types \a RndIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// The predicate operations in the parallel \a make_heap algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// executes in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a make_heap algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a make_heap algorithm returns a \a pika::future<void>
    ///           if the execution policy is of type \a task_execution_policy
    ///           and returns \a void otherwise.
    ///
    template <typename ExPolicy, typename RndIter, typename Comp>
    typename util::detail::algorithm_result<ExPolicy>::type make_heap(
        ExPolicy&& policy, RndIter first, RndIter last, Comp&& comp);

    /// Constructs a \a max \a heap in the range [first, last). Uses the
    /// operator \a < for comparisons.
    ///
    /// \note Complexity: at most (3*N) comparisons where
    ///       \a N = distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution of
    ///                     the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam RndIter     The type of the source iterators used for algorithm.
    ///                     This iterator must meet the requirements for a
    ///                     random access iterator.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of that the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     that the algorithm will be applied to.
    ///
    /// The predicate operations in the parallel \a make_heap algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// executes in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a make_heap algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a make_heap algorithm returns a \a pika::future<void>
    ///           if the execution policy is of type \a task_execution_policy
    ///           and returns \a void otherwise.
    ///
    template <typename ExPolicy, typename RndIter>
    typename pika::parallel::util::detail::algorithm_result<ExPolicy>::type
    make_heap(ExPolicy&& policy, RndIter first, RndIter last);

    // clang-format on
}    // namespace pika

#else    // DOXYGEN

#include <pika/local/config.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/functional/bind_front.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/functional/traits/is_invocable.hpp>
#include <pika/futures/future.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/parallel/util/detail/sender_util.hpp>

#include <pika/algorithms/traits/projected.hpp>
#include <pika/execution/algorithms/detail/predicates.hpp>
#include <pika/execution/executors/execution.hpp>
#include <pika/execution/executors/execution_parameters.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/parallel/algorithms/detail/dispatch.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/detail/chunk_size.hpp>
#include <pika/parallel/util/detail/handle_local_exceptions.hpp>
#include <pika/parallel/util/detail/scoped_executor_parameters.hpp>
#include <pika/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <iterator>
#include <list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika { namespace parallel { inline namespace v1 {

    //////////////////////////////////////////////////////////////////////
    // make_heap
    namespace detail {

        // Perform bottom up heap construction given a range of elements.
        // sift_down_range will take a range from [start,start-count) and
        // apply sift_down to each element in the range
        template <typename RndIter, typename Comp, typename Proj>
        void sift_down(RndIter first, Comp&& comp, Proj&& proj,
            typename std::iterator_traits<RndIter>::difference_type len,
            RndIter start)
        {
            typename std::iterator_traits<RndIter>::difference_type child =
                start - first;

            if (len < 2 || (len - 2) / 2 < child)
                return;

            child = 2 * child + 1;
            RndIter child_i = first + child;

            if ((child + 1) < len &&
                PIKA_INVOKE(comp, PIKA_INVOKE(proj, *child_i),
                    PIKA_INVOKE(proj, *(child_i + 1))))
            {
                ++child_i;
                ++child;
            }

            if (PIKA_INVOKE(
                    comp, PIKA_INVOKE(proj, *child_i), PIKA_INVOKE(proj, *start)))
                return;

            typename std::iterator_traits<RndIter>::value_type top = *start;

            do
            {
                *start = *child_i;
                start = child_i;

                if ((len - 2) / 2 < child)
                    break;

                child = 2 * child + 1;
                child_i = first + child;

                if ((child + 1) < len &&
                    PIKA_INVOKE(comp, PIKA_INVOKE(proj, *child_i),
                        PIKA_INVOKE(proj, *(child_i + 1))))
                {
                    ++child_i;
                    ++child;
                }

            } while (!PIKA_INVOKE(
                comp, PIKA_INVOKE(proj, *child_i), PIKA_INVOKE(proj, top)));

            *start = top;
        }

        template <typename RndIter, typename Comp, typename Proj>
        void sift_down_range(RndIter first, Comp&& comp, Proj&& proj,
            typename std::iterator_traits<RndIter>::difference_type len,
            RndIter start, std::size_t count)
        {
            for (std::size_t i = 0; i != count; ++i)
            {
                sift_down(first, comp, proj, len, start - i);
            }
        }

        template <typename Iter, typename Sent, typename Comp, typename Proj>
        Iter sequential_make_heap(
            Iter first, Sent last, Comp&& comp, Proj&& proj)
        {
            using difference_type =
                typename std::iterator_traits<Iter>::difference_type;

            difference_type n = last - first;
            if (n > 1)
            {
                for (difference_type start = (n - 2) / 2; start >= 0; --start)
                {
                    sift_down(first, comp, proj, n, first + start);
                }
                return first + n;
            }
            return first;
        }

        //////////////////////////////////////////////////////////////////////
        template <typename Iter>
        struct make_heap : public detail::algorithm<make_heap<Iter>, Iter>
        {
            make_heap()
              : make_heap::algorithm("make_heap")
            {
            }

            template <typename ExPolicy, typename RndIter, typename Sent,
                typename Comp, typename Proj>
            static RndIter sequential(
                ExPolicy, RndIter first, Sent last, Comp&& comp, Proj&& proj)
            {
                return sequential_make_heap(first, last,
                    PIKA_FORWARD(Comp, comp), PIKA_FORWARD(Proj, proj));
            }

            template <typename ExPolicy, typename RndIter, typename Sent,
                typename Comp, typename Proj>
            static
                typename util::detail::algorithm_result<ExPolicy, RndIter>::type
                make_heap_thread(ExPolicy&& policy, RndIter first, Sent last,
                    Comp&& comp, Proj&& proj)
            {
                typename std::iterator_traits<RndIter>::difference_type n =
                    last - first;
                if (n <= 1)
                {
                    return util::detail::algorithm_result<ExPolicy,
                        RndIter>::get(PIKA_MOVE(first));
                }

                using execution_policy = typename std::decay<ExPolicy>::type;
                using parameters_type =
                    typename execution_policy::executor_parameters_type;
                using executor_type = typename execution_policy::executor_type;

                using scoped_executor_parameters =
                    util::detail::scoped_executor_parameters_ref<
                        parameters_type, executor_type>;

                scoped_executor_parameters scoped_params(
                    policy.parameters(), policy.executor());

                std::vector<pika::future<void>> workitems;
                std::list<std::exception_ptr> errors;

                using tuple_type = pika::tuple<RndIter, std::size_t>;

                auto op = [=](tuple_type const& t) {
                    sift_down_range(first, comp, proj, (std::size_t) n,
                        pika::get<0>(t), pika::get<1>(t));
                };

                std::size_t const cores = execution::processing_units_count(
                    policy.parameters(), policy.executor());

                // Take a standard chunk size (amount of work / cores), and only
                // take a quarter of that. If our chunk size is too large a LOT
                // of the work will be done sequentially due to the level
                // barrier of heap parallelism.
                // 1/4 of the standard chunk size is an estimate to lower the
                // average number of levels done sequentially
                std::size_t chunk_size = execution::get_chunk_size(
                    policy.parameters(), policy.executor(),
                    [](std::size_t) { return 0; }, cores, n);
                chunk_size /= 4;

                std::size_t max_chunks = execution::maximal_number_of_chunks(
                    policy.parameters(), policy.executor(), cores, n);

                util::detail::adjust_chunk_size_and_max_chunks(
                    cores, n, chunk_size, max_chunks);

                try
                {
                    // Get workitems that are to be run in parallel
                    std::size_t start = (n - 2) / 2;
                    while (start > 0)
                    {
                        // Index of start of level, and amount of items in level
                        std::size_t end_exclusive =
                            (std::size_t) std::pow(
                                2, std::floor(std::log2(start))) -
                            2;
                        std::size_t level_items = (start - end_exclusive);

                        // If we can't at least run two chunks in parallel,
                        // don't bother parallelizing and simply run sequentially
                        if (chunk_size * 2 > level_items)
                        {
                            op(pika::make_tuple(first + start, level_items));
                        }
                        else
                        {
                            std::vector<tuple_type> shapes;
                            shapes.reserve(level_items / chunk_size + 1);

                            std::size_t cnt = 0;
                            while (cnt + chunk_size < level_items)
                            {
                                shapes.push_back(pika::make_tuple(
                                    first + start - cnt, chunk_size));
                                cnt += chunk_size;
                            }

                            // Schedule any left-over work
                            if (cnt < level_items)
                            {
                                shapes.push_back(pika::make_tuple(
                                    first + start - cnt, level_items - cnt));
                            }

                            // Reserve items/chunk_size spaces for async calls
                            workitems = execution::bulk_async_execute(
                                policy.executor(), op, shapes);

                            // Required synchronization per level
                            pika::wait_all_nothrow(workitems);

                            // collect exceptions
                            util::detail::handle_local_exceptions<
                                ExPolicy>::call(workitems, errors, false);
                            workitems.clear();
                        }

                        if (!errors.empty())
                            break;    // stop on errors

                        start = end_exclusive;
                    }

                    scoped_params.mark_end_of_scheduling();

                    // Perform sift down for the head node
                    sift_down(first, comp = PIKA_FORWARD(Comp, comp),
                        proj = PIKA_FORWARD(Proj, proj), n, first);
                }
                catch (...)
                {
                    util::detail::handle_local_exceptions<ExPolicy>::call(
                        std::current_exception(), errors);
                }

                // rethrow exceptions, if any
                util::detail::handle_local_exceptions<ExPolicy>::call(
                    workitems, errors);

                std::advance(first, n);
                return util::detail::algorithm_result<ExPolicy, RndIter>::get(
                    PIKA_MOVE(first));
            }

            template <typename ExPolicy, typename RndIter, typename Sent,
                typename Comp, typename Proj>
            static
                typename util::detail::algorithm_result<ExPolicy, RndIter>::type
                parallel(ExPolicy&& policy, RndIter first, Sent last,
                    Comp&& comp, Proj&& proj)
            {
                return make_heap_thread(PIKA_FORWARD(ExPolicy, policy), first,
                    last, PIKA_FORWARD(Comp, comp), PIKA_FORWARD(Proj, proj));
            }

            template <typename RndIter, typename Sent, typename Comp,
                typename Proj>
            static typename util::detail::algorithm_result<
                pika::execution::parallel_task_policy, RndIter>::type
            parallel(pika::execution::parallel_task_policy policy, RndIter first,
                Sent last, Comp&& comp, Proj&& proj)
            {
                return execution::async_execute(policy.executor(),
                    [=, comp = PIKA_FORWARD(Comp, comp),
                        proj = PIKA_FORWARD(Proj, proj)]() mutable {
                        return make_heap_thread(policy, first, last,
                            PIKA_FORWARD(Comp, comp), PIKA_FORWARD(Proj, proj));
                    });
            }
        };
    }    // namespace detail
}}}      // namespace pika::parallel::v1

namespace pika {

    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::make_heap
    inline constexpr struct make_heap_t final
      : pika::detail::tag_parallel_algorithm<make_heap_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename RndIter, typename Comp,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator<RndIter>::value &&
                pika::is_invocable_v<Comp,
                    typename std::iterator_traits<RndIter>::value_type,
                    typename std::iterator_traits<RndIter>::value_type
                >
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<
            ExPolicy>::type
        tag_fallback_invoke(make_heap_t, ExPolicy&& policy, RndIter first,
            RndIter last, Comp&& comp)
        {
            static_assert(
                pika::traits::is_random_access_iterator<RndIter>::value,
                "Requires random access iterator.");

            return pika::parallel::util::detail::algorithm_result<ExPolicy>::get(
                pika::parallel::v1::detail::make_heap<RndIter>().call(
                    PIKA_FORWARD(ExPolicy, policy), first, last,
                    PIKA_FORWARD(Comp, comp),
                    pika::parallel::util::projection_identity{}));
        }

        // clang-format off
        template <typename ExPolicy, typename RndIter,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator<RndIter>::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<
            ExPolicy>::type
        tag_fallback_invoke(
            make_heap_t, ExPolicy&& policy, RndIter first, RndIter last)
        {
            static_assert(
                pika::traits::is_random_access_iterator<RndIter>::value,
                "Requires random access iterator.");

            using value_type =
                typename std::iterator_traits<RndIter>::value_type;

            return pika::parallel::util::detail::algorithm_result<ExPolicy>::get(
                pika::parallel::v1::detail::make_heap<RndIter>().call(
                    PIKA_FORWARD(ExPolicy, policy), first, last,
                    std::less<value_type>(),
                    pika::parallel::util::projection_identity{}));
        }

        // clang-format off
        template <typename RndIter, typename Comp,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator<RndIter>::value &&
                pika::is_invocable_v<Comp,
                    typename std::iterator_traits<RndIter>::value_type,
                    typename std::iterator_traits<RndIter>::value_type
                >
            )>
        // clang-format on
        friend void tag_fallback_invoke(
            make_heap_t, RndIter first, RndIter last, Comp&& comp)
        {
            static_assert(
                pika::traits::is_random_access_iterator<RndIter>::value,
                "Requires random access iterator.");

            pika::parallel::v1::detail::make_heap<RndIter>().call(
                pika::execution::seq, first, last, PIKA_FORWARD(Comp, comp),
                pika::parallel::util::projection_identity{});
        }

        // clang-format off
        template <typename RndIter,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator<RndIter>::value
            )>
        // clang-format on
        friend void tag_fallback_invoke(
            make_heap_t, RndIter first, RndIter last)
        {
            static_assert(
                pika::traits::is_random_access_iterator<RndIter>::value,
                "Requires random access iterator.");

            using value_type =
                typename std::iterator_traits<RndIter>::value_type;

            pika::parallel::v1::detail::make_heap<RndIter>().call(
                pika::execution::seq, first, last, std::less<value_type>(),
                pika::parallel::util::projection_identity{});
        }
    } make_heap{};
}    // namespace pika

#endif    // DOXYGEN
