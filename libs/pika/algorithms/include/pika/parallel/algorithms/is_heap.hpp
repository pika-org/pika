//  Copyright (c) 2017 Taeguk Kwon
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if defined(DOXYGEN)
namespace pika {
    // clang-format off

    /// Returns whether the range is max heap. That is, true if the range is
    /// max heap, false otherwise. The function uses the given comparison
    /// function object \a comp (defaults to using operator<()).
    ///
    /// \note   Complexity:
    ///         Performs at most N applications of the comparison \a comp,
    ///         at most 2 * N applications of the projection \a proj,
    ///         where N = last - first.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam RandIter    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param comp         \a comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type \a Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that comp
    ///                     will not apply any non-constant function through the
    ///                     dereferenced iterator.
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
    /// \returns  The \a is_heap algorithm returns a \a pika::future<bool>
    ///           if the execution policy is of type \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a bool otherwise.
    ///           The \a is_heap algorithm returns whether the range is max heap.
    ///           That is, true if the range is max heap, false otherwise.
    ///
    template <typename ExPolicy, typename RandIter, typename Comp = detail::less>
    typename util::detail::algorithm_result<ExPolicy, bool>::type
    is_heap(ExPolicy&& policy, RandIter first, RandIter last,
        Comp&& comp = Comp());

    /// Returns the upper bound of the largest range beginning at \a first
    /// which is a max heap. That is, the last iterator \a it for
    /// which range [first, it) is a max heap. The function
    /// uses the given comparison function object \a comp (defaults to using
    /// operator<()).
    ///
    /// \note   Complexity:
    ///         Performs at most N applications of the comparison \a comp,
    ///         at most 2 * N applications of the projection \a proj,
    ///         where N = last - first.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam RandIter    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param comp         \a comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type \a Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that comp
    ///                     will not apply any non-constant function through the
    ///                     dereferenced iterator.
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
    /// \returns  The \a is_heap_until algorithm returns a \a pika::future<RandIter>
    ///           if the execution policy is of type \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a RandIter otherwise.
    ///           The \a is_heap_until algorithm returns the upper bound
    ///           of the largest range beginning at first which is a max heap.
    ///           That is, the last iterator \a it for which range [first, it)
    ///           is a max heap.
    ///
    template <typename ExPolicy, typename RandIter, typename Comp = detail::less>
    typename util::detail::algorithm_result<ExPolicy, RandIter>::type
    is_heap_until(ExPolicy&& policy, RandIter first, RandIter last,
        Comp&& comp = Comp());

    // clang-format on
}    // namespace pika

#else    // DOXYGEN

#include <pika/local/config.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/functional/traits/is_invocable.hpp>
#include <pika/futures/future.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/parallel/util/detail/sender_util.hpp>

#include <pika/algorithms/traits/projected.hpp>
#include <pika/execution/executors/execution.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/parallel/algorithms/detail/dispatch.hpp>
#include <pika/parallel/algorithms/detail/distance.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/loop.hpp>
#include <pika/parallel/util/partitioner.hpp>
#include <pika/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <list>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // is_heap
    namespace detail {

        // sequential is_heap with projection function
        template <typename Iter, typename Sent, typename Comp, typename Proj>
        bool sequential_is_heap(Iter first, Sent last, Comp&& comp, Proj&& proj)
        {
            typedef typename std::iterator_traits<Iter>::difference_type
                difference_type;

            difference_type count = detail::distance(first, last);

            for (difference_type i = 1; i < count; ++i)
            {
                if (PIKA_INVOKE(comp, PIKA_INVOKE(proj, *(first + (i - 1) / 2)),
                        PIKA_INVOKE(proj, *(first + i))))
                    return false;
            }
            return true;
        }

        struct is_heap_helper
        {
            template <typename ExPolicy, typename Iter, typename Sent,
                typename Comp, typename Proj>
            typename util::detail::algorithm_result<ExPolicy, bool>::type
            operator()(ExPolicy&& policy, Iter first, Sent last, Comp&& comp,
                Proj&& proj)
            {
                typedef util::detail::algorithm_result<ExPolicy, bool> result;
                typedef typename std::iterator_traits<Iter>::value_type type;
                typedef typename std::iterator_traits<Iter>::difference_type
                    difference_type;

                difference_type count = detail::distance(first, last);
                if (count <= 1)
                {
                    return result::get(true);
                }

                Iter second = first + 1;
                --count;

                util::cancellation_token<std::size_t> tok(count);

                // Note: replacing the invoke() with PIKA_INVOKE()
                // below makes gcc generate errors
                auto f1 = [tok, first, comp = PIKA_FORWARD(Comp, comp),
                              proj = PIKA_FORWARD(Proj, proj)](Iter it,
                              std::size_t part_size,
                              std::size_t base_idx) mutable -> void {
                    util::loop_idx_n<std::decay_t<ExPolicy>>(base_idx, it,
                        part_size, tok,
                        [&tok, first, &comp, &proj](
                            type const& v, std::size_t i) mutable -> void {
                            if (pika::util::invoke(comp,
                                    pika::util::invoke(proj, *(first + i / 2)),
                                    pika::util::invoke(proj, v)))
                            {
                                tok.cancel(0);
                            }
                        });
                };
                auto f2 =
                    [tok](
                        std::vector<pika::future<void>>&& data) mutable -> bool {
                    // make sure iterators embedded in function object that is
                    // attached to futures are invalidated
                    data.clear();

                    difference_type find_res =
                        static_cast<difference_type>(tok.get_data());

                    return find_res != 0;
                };

                return util::partitioner<ExPolicy, bool, void>::call_with_index(
                    PIKA_FORWARD(ExPolicy, policy), second, count, 1,
                    PIKA_MOVE(f1), PIKA_MOVE(f2));
            }
        };

        template <typename RandIter>
        struct is_heap : public detail::algorithm<is_heap<RandIter>, bool>
        {
            is_heap()
              : is_heap::algorithm("is_heap")
            {
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename Comp, typename Proj>
            static bool sequential(
                ExPolicy&&, Iter first, Sent last, Comp&& comp, Proj&& proj)
            {
                return sequential_is_heap(first, last, PIKA_FORWARD(Comp, comp),
                    PIKA_FORWARD(Proj, proj));
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename Comp, typename Proj>
            static typename util::detail::algorithm_result<ExPolicy, bool>::type
            parallel(ExPolicy&& policy, Iter first, Sent last, Comp&& comp,
                Proj&& proj)
            {
                return is_heap_helper()(PIKA_FORWARD(ExPolicy, policy), first,
                    last, PIKA_FORWARD(Comp, comp), PIKA_FORWARD(Proj, proj));
            }
        };
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename RandIter,
        typename Comp = detail::less, typename Proj = util::projection_identity,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy<ExPolicy>::value &&
            pika::traits::is_iterator<RandIter>::value &&
            traits::is_projected<Proj, RandIter>::value &&
            traits::is_indirect_callable<ExPolicy, Comp,
                traits::projected<Proj, RandIter>,
                traits::projected<Proj, RandIter>
            >::value
        )>
    // clang-format on
    PIKA_DEPRECATED_V(
        0, 1, "pika::parallel::is_heap is deprecated, use pika::is_heap instead")
        typename util::detail::algorithm_result<ExPolicy, bool>::type
        is_heap(ExPolicy&& policy, RandIter first, RandIter last,
            Comp&& comp = Comp(), Proj&& proj = Proj())
    {
        static_assert((pika::traits::is_random_access_iterator<RandIter>::value),
            "Requires a random access iterator.");

#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return detail::is_heap<RandIter>().call(PIKA_FORWARD(ExPolicy, policy),
            first, last, PIKA_FORWARD(Comp, comp), PIKA_FORWARD(Proj, proj));
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    // is_heap_until
    namespace detail {

        // sequential is_heap_until with projection function
        template <typename Iter, typename Sent, typename Comp, typename Proj>
        Iter sequential_is_heap_until(
            Iter first, Sent last, Comp&& comp, Proj&& proj)
        {
            typedef typename std::iterator_traits<Iter>::difference_type
                difference_type;

            difference_type count = detail::distance(first, last);

            for (difference_type i = 1; i < count; ++i)
            {
                if (PIKA_INVOKE(comp, PIKA_INVOKE(proj, *(first + (i - 1) / 2)),
                        PIKA_INVOKE(proj, *(first + i))))
                    return first + i;
            }
            return last;
        }

        struct is_heap_until_helper
        {
            template <typename ExPolicy, typename Iter, typename Sent,
                typename Comp, typename Proj>
            typename util::detail::algorithm_result<ExPolicy, Iter>::type
            operator()(
                ExPolicy&& policy, Iter first, Sent last, Comp comp, Proj proj)
            {
                typedef util::detail::algorithm_result<ExPolicy, Iter> result;
                typedef typename std::iterator_traits<Iter>::value_type type;
                typedef typename std::iterator_traits<Iter>::difference_type
                    difference_type;

                difference_type count = detail::distance(first, last);
                if (count <= 1)
                {
                    return result::get(PIKA_MOVE(last));
                }

                Iter second = first + 1;
                --count;

                util::cancellation_token<std::size_t> tok(count);

                // Note: replacing the invoke() with PIKA_INVOKE()
                // below makes gcc generate errors
                auto f1 = [tok, first, comp = PIKA_FORWARD(Comp, comp),
                              proj = PIKA_FORWARD(Proj, proj)](Iter it,
                              std::size_t part_size,
                              std::size_t base_idx) mutable {
                    util::loop_idx_n<std::decay_t<ExPolicy>>(base_idx, it,
                        part_size, tok,
                        [&tok, first, &comp, &proj](
                            type const& v, std::size_t i) -> void {
                            if (pika::util::invoke(comp,
                                    pika::util::invoke(proj, *(first + i / 2)),
                                    pika::util::invoke(proj, v)))
                            {
                                tok.cancel(i);
                            }
                        });
                };
                auto f2 =
                    [tok, second](
                        std::vector<pika::future<void>>&& data) mutable -> Iter {
                    // make sure iterators embedded in function object that is
                    // attached to futures are invalidated
                    data.clear();

                    difference_type find_res =
                        static_cast<difference_type>(tok.get_data());

                    std::advance(second, find_res);

                    return PIKA_MOVE(second);
                };

                return util::partitioner<ExPolicy, Iter, void>::call_with_index(
                    PIKA_FORWARD(ExPolicy, policy), second, count, 1,
                    PIKA_MOVE(f1), PIKA_MOVE(f2));
            }
        };

        template <typename RandIter>
        struct is_heap_until
          : public detail::algorithm<is_heap_until<RandIter>, RandIter>
        {
            is_heap_until()
              : is_heap_until::algorithm("is_heap_until")
            {
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename Comp, typename Proj>
            static Iter sequential(
                ExPolicy&&, Iter first, Sent last, Comp&& comp, Proj&& proj)
            {
                return sequential_is_heap_until(first, last,
                    PIKA_FORWARD(Comp, comp), PIKA_FORWARD(Proj, proj));
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename Comp, typename Proj>
            static typename util::detail::algorithm_result<ExPolicy, Iter>::type
            parallel(ExPolicy&& policy, Iter first, Sent last, Comp&& comp,
                Proj&& proj)
            {
                return is_heap_until_helper()(PIKA_FORWARD(ExPolicy, policy),
                    first, last, PIKA_FORWARD(Comp, comp),
                    PIKA_FORWARD(Proj, proj));
            }
        };
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename RandIter,
        typename Comp = detail::less, typename Proj = util::projection_identity,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy<ExPolicy>::value &&
            pika::traits::is_iterator<RandIter>::value &&
            traits::is_projected<Proj, RandIter>::value &&
            traits::is_indirect_callable<ExPolicy, Comp,
                traits::projected<Proj, RandIter>,
                traits::projected<Proj, RandIter>
            >::value
        )>
    // clang-format on
    PIKA_DEPRECATED_V(0, 1,
        "pika::parallel::is_heap_until is deprecated, use pika::is_heap_until "
        "instead")
        typename util::detail::algorithm_result<ExPolicy, RandIter>::type
        is_heap_until(ExPolicy&& policy, RandIter first, RandIter last,
            Comp&& comp = Comp(), Proj&& proj = Proj())
    {
        static_assert((pika::traits::is_random_access_iterator<RandIter>::value),
            "Requires a random access iterator.");

#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return detail::is_heap_until<RandIter>().call(
            PIKA_FORWARD(ExPolicy, policy), first, last, PIKA_FORWARD(Comp, comp),
            PIKA_FORWARD(Proj, proj));
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }
}}}    // namespace pika::parallel::v1

namespace pika {

    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::is_heap
    inline constexpr struct is_heap_t final
      : pika::detail::tag_parallel_algorithm<is_heap_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename RandIter,
            typename Comp = pika::parallel::v1::detail::less,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator<RandIter>::value &&
                pika::is_invocable_v<Comp,
                    typename std::iterator_traits<RandIter>::value_type,
                    typename std::iterator_traits<RandIter>::value_type
                >
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(is_heap_t, ExPolicy&& policy, RandIter first,
            RandIter last, Comp&& comp = Comp())
        {
            static_assert(
                (pika::traits::is_random_access_iterator<RandIter>::value),
                "Requires a random access iterator.");

            return pika::parallel::v1::detail::is_heap<RandIter>().call(
                PIKA_FORWARD(ExPolicy, policy), first, last,
                PIKA_FORWARD(Comp, comp),
                pika::parallel::util::projection_identity{});
        }

        // clang-format off
        template <typename RandIter,
            typename Comp = pika::parallel::v1::detail::less,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator<RandIter>::value &&
                pika::is_invocable_v<Comp,
                    typename std::iterator_traits<RandIter>::value_type,
                    typename std::iterator_traits<RandIter>::value_type
                >
            )>
        // clang-format on
        friend bool tag_fallback_invoke(
            is_heap_t, RandIter first, RandIter last, Comp&& comp = Comp())
        {
            static_assert(
                (pika::traits::is_random_access_iterator<RandIter>::value),
                "Requires a random access iterator.");

            return pika::parallel::v1::detail::is_heap<RandIter>().call(
                pika::execution::seq, first, last, PIKA_FORWARD(Comp, comp),
                pika::parallel::util::projection_identity{});
        }
    } is_heap{};

    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::is_heap_until
    inline constexpr struct is_heap_until_t final
      : pika::detail::tag_parallel_algorithm<is_heap_until_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename RandIter,
            typename Comp = pika::parallel::v1::detail::less,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator<RandIter>::value &&
                pika::is_invocable_v<Comp,
                    typename std::iterator_traits<RandIter>::value_type,
                    typename std::iterator_traits<RandIter>::value_type
                >
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            RandIter>::type
        tag_fallback_invoke(is_heap_until_t, ExPolicy&& policy, RandIter first,
            RandIter last, Comp&& comp = Comp())
        {
            static_assert(
                (pika::traits::is_random_access_iterator<RandIter>::value),
                "Requires a random access iterator.");

            return pika::parallel::v1::detail::is_heap_until<RandIter>().call(
                PIKA_FORWARD(ExPolicy, policy), first, last,
                PIKA_FORWARD(Comp, comp),
                pika::parallel::util::projection_identity{});
        }

        // clang-format off
        template <typename RandIter,
            typename Comp = pika::parallel::v1::detail::less,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator<RandIter>::value &&
                pika::is_invocable_v<Comp,
                    typename std::iterator_traits<RandIter>::value_type,
                    typename std::iterator_traits<RandIter>::value_type
                >
            )>
        // clang-format on
        friend RandIter tag_fallback_invoke(is_heap_until_t, RandIter first,
            RandIter last, Comp&& comp = Comp())
        {
            static_assert(
                (pika::traits::is_random_access_iterator<RandIter>::value),
                "Requires a random access iterator.");

            return pika::parallel::v1::detail::is_heap_until<RandIter>().call(
                pika::execution::seq, first, last, PIKA_FORWARD(Comp, comp),
                pika::parallel::util::projection_identity{});
        }
    } is_heap_until{};

}    // namespace pika

#endif    // DOXYGEN
