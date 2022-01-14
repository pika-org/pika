//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//  Copyright (c) 2017 Denis Blank
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file when_all.hpp

#pragma once

#if defined(DOXYGEN)
namespace pika {
    /// The function \a when_all is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after they finished executing.
    ///
    /// \param first    [in] The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_all should wait.
    /// \param last     [in] The iterator pointing to the last element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_all should wait.
    ///
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to \a when_all.
    ///           - future<Container<future<R>>>: If the input cardinality is
    ///             unknown at compile time and the futures are all of the
    ///             same type. The order of the futures in the output container
    ///             will be the same as given by the input iterator.
    ///
    /// \note Calling this version of \a when_all where first == last, returns
    ///       a future with an empty container that is immediately ready.
    ///       Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a when_all will not throw an exception,
    ///       but the futures held in the output collection may.
    template <typename InputIter,
        typename Container = vector<
            future<typename std::iterator_traits<InputIter>::value_type>>>
    pika::future<Container> when_all(InputIter first, InputIter last);

    /// The function \a when_all is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after they finished executing.
    ///
    /// \param values   [in] A range holding an arbitrary amount of \a future
    ///                 or \a shared_future objects for which \a when_all
    ///                 should wait.
    ///
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to when_all.
    ///           - future<Container<future<R>>>: If the input cardinality is
    ///             unknown at compile time and the futures are all of the
    ///             same type.
    ///
    /// \note Calling this version of \a when_all where the input container is
    ///       empty, returns a future with an empty container that is immediately
    ///       ready.
    ///       Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a when_all will not throw an exception,
    ///       but the futures held in the output collection may.
    template <typename Range>
    pika::future<Range> when_all(Range&& values);

    /// The function \a when_all is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after they finished executing.
    ///
    /// \param futures  [in] An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a when_all should wait.
    ///
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to \a when_all.
    ///           - future<tuple<future<T0>, future<T1>, future<T2>...>>: If
    ///             inputs are fixed in number and are of heterogeneous types.
    ///             The inputs can be any arbitrary number of future objects.
    ///           - future<tuple<>> if \a when_all is called with zero arguments.
    ///             The returned future will be initially ready.
    ///
    /// \note Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a when_all will not throw an exception,
    ///       but the futures held in the output collection may.
    template <typename... T>
    pika::future<pika::tuple<pika::future<T>...>> when_all(T&&... futures);

    /// The function \a when_all_n is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after they finished executing.
    ///
    /// \param begin    [in] The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_all_n should wait.
    /// \param count    [in] The number of elements in the sequence starting at
    ///                 \a first.
    ///
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to \a when_all_n.
    ///           - future<Container<future<R>>>: If the input cardinality is
    ///             unknown at compile time and the futures are all of the
    ///             same type. The order of the futures in the output vector
    ///             will be the same as given by the input iterator.
    ///
    /// \throws This function will throw errors which are encountered while
    ///         setting up the requested operation only. Errors encountered
    ///         while executing the operations delivering the results to be
    ///         stored in the futures are reported through the futures
    ///         themselves.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a pika::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           pika::exception.
    ///
    /// \note     None of the futures in the input sequence are invalidated.
    template <typename InputIter,
        typename Container = vector<
            future<typename std::iterator_traits<InputIter>::value_type>>>
    pika::future<Container> when_all_n(InputIter begin, std::size_t count);
}    // namespace pika

#else    // DOXYGEN

#include <pika/local/config.hpp>
#include <pika/allocator_support/internal_allocator.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/futures/detail/future_data.hpp>
#include <pika/futures/detail/future_transforms.hpp>
#include <pika/futures/future.hpp>
#include <pika/futures/traits/acquire_future.hpp>
#include <pika/futures/traits/acquire_shared_state.hpp>
#include <pika/futures/traits/future_access.hpp>
#include <pika/futures/traits/future_traits.hpp>
#include <pika/futures/traits/is_future.hpp>
#include <pika/futures/traits/is_future_range.hpp>
#include <pika/pack_traversal/pack_traversal_async.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace pika {

    namespace lcos { namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Enable = void>
        struct when_all_result
        {
            using type = T;

            static type call(T&& t) noexcept
            {
                return PIKA_MOVE(t);
            }
        };

        template <typename T>
        struct when_all_result<pika::tuple<T>,
            std::enable_if_t<pika::traits::is_future_range_v<T>>>
        {
            using type = T;

            static type call(pika::tuple<T>&& t) noexcept
            {
                return PIKA_MOVE(pika::get<0>(t));
            }
        };

        template <typename T>
        using when_all_result_t = typename when_all_result<T>::type;

        template <typename Tuple>
        class async_when_all_frame
          : public future_data<when_all_result_t<Tuple>>
        {
        public:
            using result_type = when_all_result_t<Tuple>;
            using type = pika::future<result_type>;
            using base_type = pika::lcos::detail::future_data<result_type>;

            explicit async_when_all_frame(
                typename base_type::init_no_addref no_addref)
              : base_type(no_addref)
            {
            }

            template <typename T>
            auto operator()(pika::util::async_traverse_visit_tag, T&& current)
                -> decltype(async_visit_future(PIKA_FORWARD(T, current)))
            {
                return async_visit_future(PIKA_FORWARD(T, current));
            }

            template <typename T, typename N>
            auto operator()(
                pika::util::async_traverse_detach_tag, T&& current, N&& next)
                -> decltype(async_detach_future(
                    PIKA_FORWARD(T, current), PIKA_FORWARD(N, next)))
            {
                return async_detach_future(
                    PIKA_FORWARD(T, current), PIKA_FORWARD(N, next));
            }

            template <typename T>
            void operator()(pika::util::async_traverse_complete_tag, T&& pack)
            {
                this->set_value(
                    when_all_result<Tuple>::call(PIKA_FORWARD(T, pack)));
            }
        };

        template <typename... T>
        typename async_when_all_frame<
            pika::tuple<pika::traits::acquire_future_t<T>...>>::type
        when_all_impl(T&&... args)
        {
            using result_type = pika::tuple<pika::traits::acquire_future_t<T>...>;
            using frame_type = async_when_all_frame<result_type>;
            using no_addref = typename frame_type::base_type::init_no_addref;

            auto frame = pika::util::traverse_pack_async_allocator(
                pika::util::internal_allocator<>{},
                pika::util::async_traverse_in_place_tag<frame_type>{},
                no_addref{},
                pika::traits::acquire_future_disp()(PIKA_FORWARD(T, args))...);

            return pika::traits::future_access<
                typename frame_type::type>::create(PIKA_MOVE(frame));
        }
    }}    // namespace lcos::detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Args>
    auto when_all(Args&&... args) -> decltype(
        pika::lcos::detail::when_all_impl(PIKA_FORWARD(Args, args)...))
    {
        return pika::lcos::detail::when_all_impl(PIKA_FORWARD(Args, args)...);
    }

    template <typename Iterator,
        typename Container =
            std::vector<pika::lcos::detail::future_iterator_traits_t<Iterator>>,
        typename Enable =
            std::enable_if_t<pika::traits::is_iterator_v<Iterator>>>
    decltype(auto) when_all(Iterator begin, Iterator end)
    {
        return pika::lcos::detail::when_all_impl(
            pika::lcos::detail::acquire_future_iterators<Iterator, Container>(
                begin, end));
    }

    template <typename Iterator,
        typename Container =
            std::vector<pika::lcos::detail::future_iterator_traits_t<Iterator>>,
        typename Enable =
            std::enable_if_t<pika::traits::is_iterator_v<Iterator>>>
    decltype(auto) when_all_n(Iterator begin, std::size_t count)
    {
        return pika::lcos::detail::when_all_impl(
            pika::lcos::detail::acquire_future_n<Iterator, Container>(
                begin, count));
    }

    inline pika::future<pika::tuple<>>    //-V524
    when_all()
    {
        return pika::make_ready_future(pika::tuple<>());
    }
}    // namespace pika

namespace pika::lcos {

    template <typename... Args>
    PIKA_DEPRECATED_V(
        0, 1, "pika::lcos::when_all is deprecated. Use pika::when_all instead.")
    auto when_all(Args&&... args)
    {
        return pika::when_all(PIKA_FORWARD(Args, args)...);
    }

    template <typename Iterator,
        typename Enable =
            std::enable_if_t<pika::traits::is_iterator_v<Iterator>>>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::when_all_n is deprecated. Use pika::when_all_n instead.")
    auto when_all_n(Iterator begin, std::size_t count)
    {
        return pika::when_all(begin, count);
    }
}    // namespace pika::lcos

#endif    // DOXYGEN
