//  Copyright (c) 2017 Taeguk Kwon
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/unique.hpp

#pragma once

#if defined(DOXYGEN)

namespace pika { namespace ranges {
    // clang-format off

    ///////////////////////////////////////////////////////////////////////////
    /// Eliminates all but the first element from every consecutive group of
    /// equivalent elements from the range [first, last) and returns a
    /// past-the-end iterator for the new logical end of the range.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first - 1 applications of
    ///         the predicate \a pred and no more than twice as many
    ///         applications of the projection \a proj.
    ///
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a unique requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     binary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be
    ///                     such that objects of types \a FwdIter can be
    ///                     dereferenced and then implicitly converted to
    ///                     both \a Type1 and \a Type2
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a unique algorithm invoked without
    /// an execution policy object execute in sequential order in the
    /// calling thread.
    ///
    /// \returns  The \a unique algorithm returns \a subrange_t<FwdIter, Sent>.
    ///           The \a unique algorithm returns an object {ret, last},
    ///           where ret is a past-the-end iterator for a new
    ///           subrange.
    ///
    template <typename FwdIter, typename Sent, typename Pred, typename Proj>
    subrange_t<FwdIter, Sent> unique(FwdIter first, Sent last, Pred&& pred,
        Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Eliminates all but the first element from every consecutive group of
    /// equivalent elements from the range [first, last) and returns a
    /// past-the-end iterator for the new logical end of the range.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first - 1 applications of
    ///         the predicate \a pred and no more than twice as many
    ///         applications of the projection \a proj.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a unique requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     binary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be
    ///                     such that objects of types \a FwdIter can be
    ///                     dereferenced and then implicitly converted to
    ///                     both \a Type1 and \a Type2
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a unique algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a unique algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a unique algorithm returns \a subrange_t<FwdIter, Sent>.
    ///           The \a unique algorithm returns an object {ret, last},
    ///           where ret is a past-the-end iterator for a new
    ///           subrange.
    ///
    template <typename ExPolicy, typename FwdIter, typename Sent, typename Pred,
        typename Proj>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        subrange_t<FwdIter, Sent>>::type
    unique(ExPolicy&& policy, FwdIter first, Sent last, Pred&& pred,
        Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Eliminates all but the first element from every consecutive group of
    /// equivalent elements from the range \a rng and returns a
    /// past-the-end iterator for the new logical end of the range.
    ///
    /// \note   Complexity: Performs not more than N assignments,
    ///         exactly N - 1 applications of the predicate \a pred and
    ///         no more than twice as many applications of the projection
    ///         \a proj, where N = std::distance(begin(rng), end(rng)).
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a unique requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     binary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a unique algorithm invoked without
    /// an execution policy object execute in sequential order in the
    /// calling thread.
    ///
    /// \returns  The \a unique algorithm returns
    ///           \a subrange_t<typename pika::traits::range_iterator<Rng>
    ///           ::type,pika::traits::range_iterator_t<Rng>>.
    ///           The \a unique algorithm returns an object {ret, last},
    ///           where ret is a past-the-end iterator for a new
    ///           subrange.
    ///
    template <typename Rng, typename Pred, typename Proj>
    subrange_t<pika::traits::range_iterator_t<Rng>,
        pika::traits::range_iterator_t<Rng>>
    unique(Rng&& rng, Pred&& pred, Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Eliminates all but the first element from every consecutive group of
    /// equivalent elements from the range \a rng and returns a
    /// past-the-end iterator for the new logical end of the range.
    ///
    /// \note   Complexity: Performs not more than N assignments,
    ///         exactly N - 1 applications of the predicate \a pred and
    ///         no more than twice as many applications of the projection
    ///         \a proj, where N = std::distance(begin(rng), end(rng)).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a unique requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     binary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a unique algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a unique algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a unique algorithm returns a \a pika::future
    ///           <subrange_t<pika::traits::range_iterator_t<Rng>,
    ///           pika::traits::range_iterator_t<Rng>>>
    ///           if the execution policy is of type \a sequenced_task_policy
    ///           or \a parallel_task_policy and returns
    ///           \a subrange_t<typename pika::traits::range_iterator<Rng>
    ///           ::type,pika::traits::range_iterator_t<Rng>>.
    ///           otherwise.
    ///           The \a unique algorithm returns an object {ret, last},
    ///           where ret is a past-the-end iterator for a new
    ///           subrange.
    ///
    template <typename ExPolicy, typename Rng, typename Pred, typename Proj>
    typename util::detail::algorithm_result<ExPolicy,
    subrange_t<pika::traits::range_iterator_t<Rng>,
    pika::traits::range_iterator_t<Rng>>::type
    unique(ExPolicy&& policy, Rng&& rng, Pred&& pred, Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the elements from the range [first, last),
    /// to another range beginning at \a dest in such a way that
    /// there are no consecutive equal elements. Only the first element of
    /// each group of equal elements is copied.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first - 1 applications of
    ///         the predicate \a pred and no more than twice as many
    ///         applications of the projection \a proj
    ///
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a unique_copy requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     binary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a unique_copy algorithm invoked
    /// without an execution policy object  will execute in sequential
    /// order in the calling thread.
    ///
    /// \returns  The \a unique_copy algorithm returns a
    ///           returns unique_copy_result<FwdIter, OutIter>.
    ///           The \a unique_copy algorithm returns an in_out_result with
    ///           the source iterator to one past the last element and out
    ///           containing the destination iterator to the end of the
    ///           \a dest range.
    ///
    template <typename InIter, typename Sent, typename OutIter,
        typename Pred, typename Proj>
    unique_copy_result<FwdIter, OutIter> unique_copy(InIter first,
        Sent last, OutIter dest, Pred&& pred, Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the elements from the range [first, last),
    /// to another range beginning at \a dest in such a way that
    /// there are no consecutive equal elements. Only the first element of
    /// each group of equal elements is copied.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first - 1 applications of
    ///         the predicate \a pred and no more than twice as many
    ///         applications of the projection \a proj
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter1.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a unique_copy requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     binary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a unique_copy algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a unique_copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a unique_copy algorithm returns areturns pika::future<
    ///           unique_copy_result<FwdIter1, FwdIter2>> if the
    ///           execution policy is of type \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a
    ///           unique_copy_result<FwdIter1, FwdIter2> otherwise.
    ///           The \a unique_copy algorithm returns an in_out_result with
    ///           the source iterator to one past the last element and out
    ///           containing the destination iterator to the end of the
    ///           \a dest range.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Sent,
        typename FwdIter2, typename Pred, typename Proj>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        unique_copy_result<FwdIter1, FwdIter2>>::type
    unique_copy(ExPolicy&& policy, FwdIter1 first, Sent last,
        FwdIter2 dest, Pred&& pred, Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the elements from the range \a rng,
    /// to another range beginning at \a dest in such a way that
    /// there are no consecutive equal elements. Only the first element of
    /// each group of equal elements is copied.
    ///
    /// \note   Complexity: Performs not more than N assignments,
    ///         exactly N - 1 applications of the predicate \a pred,
    ///         where N = std::distance(begin(rng), end(rng)).
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam O           The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a unique_copy requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by the range \a rng. This is an
    ///                     binary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a unique_copy algorithm invoked
    /// without an execution policy object  will execute in sequential
    /// order in the calling thread.
    ///
    /// \returns  The \a unique_copy algorithm returns \a
    ///           unique_copy_result<
    ///           pika::traits::range_iterator_t<Rng>, O>.
    ///           The \a unique_copy algorithm returns the pair of
    ///           the source iterator to \a last, and
    ///           the destination iterator to the end of the \a dest range.
    ///
    template <typename Rng, typename O, typename Pred, typename Proj>
    unique_copy_result<pika::traits::range_iterator_t<Rng>, O>
    unique_copy(Rng&& rng, O dest, Pred&& pred, Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the elements from the range \a rng,
    /// to another range beginning at \a dest in such a way that
    /// there are no consecutive equal elements. Only the first element of
    /// each group of equal elements is copied.
    ///
    /// \note   Complexity: Performs not more than N assignments,
    ///         exactly N - 1 applications of the predicate \a pred,
    ///         where N = std::distance(begin(rng), end(rng)).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam O           The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a unique_copy requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by the range \a rng. This is an
    ///                     binary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a unique_copy algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a unique_copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a unique_copy algorithm returns a
    ///           \a pika::future<unique_copy_result<
    ///           pika::traits::range_iterator_t<Rng>, O>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a unique_copy_result<
    ///           pika::traits::range_iterator_t<Rng>, O>
    ///           otherwise.
    ///           The \a unique_copy algorithm returns the pair of
    ///           the source iterator to \a last, and
    ///           the destination iterator to the end of the \a dest range.
    ///
    template <typename InIter, typename Sent, typename OutIter,
        typename Pred, typename Proj>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        unique_copy_result<pika::traits::range_iterator_t<Rng>,
        O>>::type
    unique_copy(ExPolicy&& policy, Rng&& rng, O dest,
        Pred&& pred, Proj&& proj);

    // clang-format on
}}    // namespace pika::ranges

#else

#include <pika/local/config.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/execution/algorithms/detail/predicates.hpp>
#include <pika/iterator_support/iterator_range.hpp>
#include <pika/iterator_support/range.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/iterator_support/traits/is_range.hpp>

#include <pika/algorithms/traits/projected.hpp>
#include <pika/algorithms/traits/projected_range.hpp>
#include <pika/parallel/algorithms/unique.hpp>
#include <pika/parallel/util/detail/sender_util.hpp>

#include <type_traits>
#include <utility>

namespace pika { namespace parallel { inline namespace v1 {
    // clang-format off
    template <typename ExPolicy, typename Rng,
        typename Pred = detail::equal_to,
        typename Proj = util::projection_identity,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy<ExPolicy>::value &&
            pika::traits::is_range<Rng>::value &&
            traits::is_projected_range<Proj, Rng>::value &&
            traits::is_indirect_callable<ExPolicy, Pred,
                traits::projected_range<Proj, Rng>,
                traits::projected_range<Proj, Rng>>::value
        )>
    // clang-format on
    PIKA_DEPRECATED_V(
        0, 1, "pika::parallel::unique is deprecated, use pika::unique instead")
        typename util::detail::algorithm_result<ExPolicy,
            pika::traits::range_iterator_t<Rng>>::type unique(ExPolicy&& policy,
            Rng&& rng, Pred&& pred = Pred(), Proj&& proj = Proj())
    {
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return unique(PIKA_FORWARD(ExPolicy, policy), pika::util::begin(rng),
            pika::util::end(rng), PIKA_FORWARD(Pred, pred),
            PIKA_FORWARD(Proj, proj));
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

    // clang-format off
    template <typename ExPolicy, typename Rng, typename FwdIter2,
        typename Pred = detail::equal_to,
        typename Proj = util::projection_identity,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy<ExPolicy>::value &&
            pika::traits::is_range<Rng>::value &&
            pika::traits::is_iterator<FwdIter2>::value &&
            traits::is_projected_range<Proj, Rng>::value &&
            traits::is_indirect_callable<ExPolicy, Pred,
                traits::projected_range<Proj, Rng>,
                traits::projected_range<Proj, Rng>>::value
        )>
    // clang-format on
    PIKA_DEPRECATED_V(0, 1,
        "pika::parallel::unique_copy is deprecated, use pika::unique_copy "
        "instead") typename util::detail::algorithm_result<ExPolicy,
        util::in_out_result<pika::traits::range_iterator_t<Rng>, FwdIter2>>::type
        unique_copy(ExPolicy&& policy, Rng&& rng, FwdIter2 dest,
            Pred&& pred = Pred(), Proj&& proj = Proj())
    {
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return unique_copy(PIKA_FORWARD(ExPolicy, policy), pika::util::begin(rng),
            pika::util::end(rng), dest, PIKA_FORWARD(Pred, pred),
            PIKA_FORWARD(Proj, proj));
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }
}}}    // namespace pika::parallel::v1

namespace pika { namespace ranges {

    ///////////////////////////////////////////////////////////////////////////
    template <typename I, typename S>
    using subrange_t = pika::util::iterator_range<I, S>;

    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::ranges::unique
    inline constexpr struct unique_t final
      : pika::detail::tag_parallel_algorithm<unique_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Sent,
            typename Pred = ranges::equal_to,
            typename Proj = parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator_v<FwdIter> &&
                pika::traits::is_sentinel_for<Sent, FwdIter>::value &&
                parallel::traits::is_projected<Proj, FwdIter>::value &&
                parallel::traits::is_indirect_callable<
                    pika::execution::sequenced_policy, Pred,
                    parallel::traits::projected<Proj, FwdIter>,
                    parallel::traits::projected<Proj, FwdIter>>::value
            )>
        // clang-format on
        friend subrange_t<FwdIter, Sent> tag_fallback_invoke(
            pika::ranges::unique_t, FwdIter first, Sent last,
            Pred&& pred = Pred(), Proj&& proj = Proj())
        {
            static_assert(pika::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return pika::parallel::util::make_subrange<FwdIter, Sent>(
                pika::parallel::v1::detail::unique<FwdIter>().call(
                    pika::execution::seq, first, last, PIKA_FORWARD(Pred, pred),
                    PIKA_FORWARD(Proj, proj)),
                last);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename Pred = ranges::equal_to,
            typename Proj = parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator_v<FwdIter> &&
                pika::traits::is_sentinel_for<Sent, FwdIter>::value &&
                parallel::traits::is_projected<Proj, FwdIter>::value &&
                parallel::traits::is_indirect_callable<
                    ExPolicy, Pred,
                    parallel::traits::projected<Proj, FwdIter>,
                    parallel::traits::projected<Proj, FwdIter>>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            subrange_t<FwdIter, Sent>>::type
        tag_fallback_invoke(pika::ranges::unique_t, ExPolicy&& policy,
            FwdIter first, Sent last, Pred&& pred = Pred(),
            Proj&& proj = Proj())
        {
            static_assert(pika::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return pika::parallel::util::make_subrange<FwdIter, Sent>(
                pika::parallel::v1::detail::unique<FwdIter>().call(
                    PIKA_FORWARD(ExPolicy, policy), first, last,
                    PIKA_FORWARD(Pred, pred), PIKA_FORWARD(Proj, proj)),
                last);
        }

        // clang-format off
        template <typename Rng,
            typename Pred = ranges::equal_to,
            typename Proj = parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_range<Rng>::value &&
                pika::parallel::traits::is_projected_range<Proj, Rng>::value &&
                pika::parallel::traits::is_indirect_callable<
                    pika::execution::sequenced_policy, Pred,
                    pika::parallel::traits::projected_range<Proj, Rng>,
                    pika::parallel::traits::projected_range<Proj, Rng>>::value
            )>
        // clang-format on
        friend subrange_t<pika::traits::range_iterator_t<Rng>,
            pika::traits::range_iterator_t<Rng>>
        tag_fallback_invoke(pika::ranges::unique_t, Rng&& rng,
            Pred&& pred = Pred(), Proj&& proj = Proj())
        {
            using iterator_type = pika::traits::range_iterator_t<Rng>;

            static_assert(pika::traits::is_forward_iterator_v<iterator_type>,
                "Requires at least input iterator.");

            return pika::parallel::util::make_subrange<
                pika::traits::range_iterator_t<Rng>,
                typename pika::traits::range_sentinel<Rng>::type>(
                pika::parallel::v1::detail::unique<
                    pika::traits::range_iterator_t<Rng>>()
                    .call(pika::execution::seq, pika::util::begin(rng),
                        pika::util::end(rng), PIKA_FORWARD(Pred, pred),
                        PIKA_FORWARD(Proj, proj)),
                pika::util::end(rng));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
            typename Pred = ranges::equal_to,
            typename Proj = parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_range<Rng>::value &&
                pika::parallel::traits::is_projected_range<Proj, Rng>::value &&
                pika::parallel::traits::is_indirect_callable<
                    ExPolicy, Pred,
                    pika::parallel::traits::projected_range<Proj, Rng>,
                    pika::parallel::traits::projected_range<Proj, Rng>>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            subrange_t<pika::traits::range_iterator_t<Rng>,
                pika::traits::range_iterator_t<Rng>>>::type
        tag_fallback_invoke(pika::ranges::unique_t, ExPolicy&& policy, Rng&& rng,
            Pred&& pred = Pred(), Proj&& proj = Proj())
        {
            using iterator_type = pika::traits::range_iterator_t<Rng>;

            static_assert(pika::traits::is_forward_iterator_v<iterator_type>,
                "Requires at least forward iterator.");

            return pika::parallel::util::make_subrange<
                pika::traits::range_iterator_t<Rng>,
                typename pika::traits::range_sentinel<Rng>::type>(
                pika::parallel::v1::detail::unique<
                    pika::traits::range_iterator_t<Rng>>()
                    .call(PIKA_FORWARD(ExPolicy, policy), pika::util::begin(rng),
                        pika::util::end(rng), PIKA_FORWARD(Pred, pred),
                        PIKA_FORWARD(Proj, proj)),
                pika::util::end(rng));
        }
    } unique{};

    template <typename I, typename O>
    using unique_copy_result = parallel::util::in_out_result<I, O>;

    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::ranges::unique_copy
    inline constexpr struct unique_copy_t final
      : pika::detail::tag_parallel_algorithm<unique_copy_t>
    {
    private:
        // clang-format off
        template <typename InIter, typename Sent, typename O,
            typename Pred = ranges::equal_to,
            typename Proj = parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator_v<InIter> &&
                pika::traits::is_sentinel_for<Sent, InIter>::value &&
                parallel::traits::is_projected<Proj, InIter>::value &&
                parallel::traits::is_indirect_callable<
                    pika::execution::sequenced_policy, Pred,
                    parallel::traits::projected<Proj, InIter>,
                    parallel::traits::projected<Proj, InIter>>::value
            )>
        // clang-format on
        friend unique_copy_result<InIter, O> tag_fallback_invoke(
            pika::ranges::unique_copy_t, InIter first, Sent last, O dest,
            Pred&& pred = Pred(), Proj&& proj = Proj())
        {
            static_assert(pika::traits::is_input_iterator_v<InIter>,
                "Requires at least input iterator.");

            using result_type = unique_copy_result<InIter, O>;

            return pika::parallel::v1::detail::unique_copy<result_type>().call(
                pika::execution::seq, first, last, dest, PIKA_FORWARD(Pred, pred),
                PIKA_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename O,
            typename Pred = ranges::equal_to,
            typename Proj = parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator_v<FwdIter> &&
                pika::traits::is_sentinel_for<Sent, FwdIter>::value &&
                parallel::traits::is_projected<Proj, FwdIter>::value &&
                parallel::traits::is_indirect_callable<
                    ExPolicy, Pred,
                    parallel::traits::projected<Proj, FwdIter>,
                    parallel::traits::projected<Proj, FwdIter>>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            unique_copy_result<FwdIter, O>>::type
        tag_fallback_invoke(pika::ranges::unique_copy_t, ExPolicy&& policy,
            FwdIter first, Sent last, O dest, Pred&& pred = Pred(),
            Proj&& proj = Proj())
        {
            static_assert(pika::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            using result_type = unique_copy_result<FwdIter, O>;

            return pika::parallel::v1::detail::unique_copy<result_type>().call(
                PIKA_FORWARD(ExPolicy, policy), first, last, dest,
                PIKA_FORWARD(Pred, pred), PIKA_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename Rng, typename O,
            typename Pred = ranges::equal_to,
            typename Proj = parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_range<Rng>::value &&
                pika::parallel::traits::is_projected_range<Proj, Rng>::value &&
                pika::parallel::traits::is_indirect_callable<
                    pika::execution::sequenced_policy, Pred,
                    pika::parallel::traits::projected_range<Proj, Rng>,
                    pika::parallel::traits::projected_range<Proj, Rng>>::value
            )>
        // clang-format on
        friend unique_copy_result<pika::traits::range_iterator_t<Rng>, O>
        tag_fallback_invoke(pika::ranges::unique_copy_t, Rng&& rng, O dest,
            Pred&& pred = Pred(), Proj&& proj = Proj())
        {
            using iterator_type = pika::traits::range_iterator_t<Rng>;

            static_assert(pika::traits::is_input_iterator_v<iterator_type>,
                "Requires at least input iterator.");

            using result_type = unique_copy_result<iterator_type, O>;

            return pika::parallel::v1::detail::unique_copy<result_type>().call(
                pika::execution::seq, pika::util::begin(rng), pika::util::end(rng),
                dest, PIKA_FORWARD(Pred, pred), PIKA_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename O,
            typename Pred = ranges::equal_to,
            typename Proj = parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_range<Rng>::value &&
                pika::parallel::traits::is_projected_range<Proj, Rng>::value &&
                pika::parallel::traits::is_indirect_callable<
                    ExPolicy, Pred,
                    pika::parallel::traits::projected_range<Proj, Rng>,
                    pika::parallel::traits::projected_range<Proj, Rng>>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            unique_copy_result<pika::traits::range_iterator_t<Rng>, O>>::type
        tag_fallback_invoke(pika::ranges::unique_copy_t, ExPolicy&& policy,
            Rng&& rng, O dest, Pred&& pred = Pred(), Proj&& proj = Proj())
        {
            using iterator_type = pika::traits::range_iterator_t<Rng>;

            static_assert(pika::traits::is_forward_iterator_v<iterator_type>,
                "Requires at least input iterator.");

            using result_type = unique_copy_result<iterator_type, O>;

            return pika::parallel::v1::detail::unique_copy<result_type>().call(
                PIKA_FORWARD(ExPolicy, policy), pika::util::begin(rng),
                pika::util::end(rng), dest, PIKA_FORWARD(Pred, pred),
                PIKA_FORWARD(Proj, proj));
        }
    } unique_copy{};
}}    // namespace pika::ranges

#endif
