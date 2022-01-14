//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//  Copyright (c) 2016 Lukas Troska
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file when_each.hpp

#pragma once

#if defined(DOXYGEN)
namespace pika {
    /// The function \a when_each is an operator allowing to join on the results
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the event of all those futures
    /// having finished executing. It also calls the supplied callback
    /// for each of the futures which becomes ready.
    ///
    /// \param f        The function which will be called for each of the
    ///                 input futures once the future has become ready.
    ///
    /// \param futures  A vector holding an arbitrary amount of \a future or
    ///                 \a shared_future objects for which \a wait_each should
    ///                 wait.
    ///
    /// \note This function consumes the futures as they are passed on to the
    ///       supplied function. The callback should take one or two parameters,
    ///       namely either a \a future to be processed or a type that
    ///       \a std::size_t is implicitly convertible to as the
    ///       first parameter and the \a future as the second
    ///       parameter. The first parameter will correspond to the
    ///       index of the current \a future in the collection.
    ///
    /// \return   Returns a future representing the event of all input futures
    ///           being ready.
    ///
    template <typename F, typename Future>
    future<void> when_each(F&& f, std::vector<Future>&& futures);

    /// The function \a when_each is an operator allowing to join on the results
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the event of all those futures
    /// having finished executing. It also calls the supplied callback
    /// for each of the futures which becomes ready.
    ///
    /// \param f        The function which will be called for each of the
    ///                 input futures once the future has become ready.
    /// \param begin    The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_each should wait.
    /// \param end      The iterator pointing to the last element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_each should wait.
    ///
    /// \note This function consumes the futures as they are passed on to the
    ///       supplied function. The callback should take one or two parameters,
    ///       namely either a \a future to be processed or a type that
    ///       \a std::size_t is implicitly convertible to as the
    ///       first parameter and the \a future as the second
    ///       parameter. The first parameter will correspond to the
    ///       index of the current \a future in the collection.
    ///
    /// \return   Returns a future representing the event of all input futures
    ///           being ready.
    ///
    template <typename F, typename Iterator>
    future<Iterator> when_each(F&& f, Iterator begin, Iterator end);

    /// The function \a when_each is an operator allowing to join on the results
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the event of all those futures
    /// having finished executing. It also calls the supplied callback
    /// for each of the futures which becomes ready.
    ///
    /// \param f        The function which will be called for each of the
    ///                 input futures once the future has become ready.
    /// \param futures  An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a wait_each should wait.
    ///
    /// \note This function consumes the futures as they are passed on to the
    ///       supplied function. The callback should take one or two parameters,
    ///       namely either a \a future to be processed or a type that
    ///       \a std::size_t is implicitly convertible to as the
    ///       first parameter and the \a future as the second
    ///       parameter. The first parameter will correspond to the
    ///       index of the current \a future in the collection.
    ///
    /// \return   Returns a future representing the event of all input futures
    ///           being ready.
    ///
    template <typename F, typename... Ts>
    future<void> when_each(F&& f, Ts&&... futures);

    /// The function \a when_each is an operator allowing to join on the results
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the event of all those futures
    /// having finished executing. It also calls the supplied callback
    /// for each of the futures which becomes ready.
    ///
    /// \param f        The function which will be called for each of the
    ///                 input futures once the future has become ready.
    /// \param begin    The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_each_n should wait.
    /// \param count    The number of elements in the sequence starting at
    ///                 \a first.
    ///
    /// \note This function consumes the futures as they are passed on to the
    ///       supplied function. The callback should take one or two parameters,
    ///       namely either a \a future to be processed or a type that
    ///       \a std::size_t is implicitly convertible to as the
    ///       first parameter and the \a future as the second
    ///       parameter. The first parameter will correspond to the
    ///       index of the current \a future in the collection.
    ///
    /// \return   Returns a future holding the iterator pointing to the first
    ///           element after the last one.
    ///
    template <typename F, typename Iterator>
    future<Iterator> when_each_n(F&& f, Iterator begin, std::size_t count);
}    // namespace pika

#else    // DOXYGEN

#include <pika/local/config.hpp>
#include <pika/async_base/launch_policy.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/futures/future.hpp>
#include <pika/futures/traits/acquire_future.hpp>
#include <pika/futures/traits/detail/future_traits.hpp>
#include <pika/futures/traits/future_access.hpp>
#include <pika/futures/traits/future_traits.hpp>
#include <pika/futures/traits/is_future.hpp>
#include <pika/futures/traits/is_future_range.hpp>
#include <pika/iterator_support/range.hpp>
#include <pika/modules/memory.hpp>
#include <pika/type_support/decay.hpp>
#include <pika/type_support/unwrap_ref.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace pika {

    namespace lcos { namespace detail {

        template <typename Tuple, typename F>
        struct when_each_frame    //-V690
          : lcos::detail::future_data<void>
        {
            using type = pika::future<void>;

        private:
            when_each_frame(when_each_frame const&) = delete;
            when_each_frame(when_each_frame&&) = delete;

            when_each_frame& operator=(when_each_frame const&) = delete;
            when_each_frame& operator=(when_each_frame&&) = delete;

            template <std::size_t I>
            struct is_end
              : std::integral_constant<bool, pika::tuple_size<Tuple>::value == I>
            {
            };

            template <std::size_t I>
            static constexpr bool is_end_v = is_end<I>::value;

        public:
            template <typename Tuple_, typename F_>
            when_each_frame(Tuple_&& t, F_&& f, std::size_t needed_count)
              : t_(PIKA_FORWARD(Tuple_, t))
              , f_(PIKA_FORWARD(F_, f))
              , count_(0)
              , needed_count_(needed_count)
            {
            }

        public:
            template <std::size_t I>
            PIKA_FORCEINLINE void do_await()
            {
                if constexpr (is_end_v<I>)
                {
                    this->set_value(util::unused);
                }
                else
                {
                    using future_type = pika::util::decay_unwrap_t<
                        typename pika::tuple_element<I, Tuple>::type>;

                    if constexpr (pika::traits::is_future_v<future_type> ||
                        pika::traits::is_ref_wrapped_future_v<future_type>)
                    {
                        await_future<I>();
                    }
                    else
                    {
                        static_assert(
                            pika::traits::is_future_range_v<future_type> ||
                                pika::traits::is_ref_wrapped_future_range_v<
                                    future_type>,
                            "element must be future or range of futures");

                        auto&& curr = pika::util::unwrap_ref(pika::get<I>(t_));
                        await_range<I>(
                            pika::util::begin(curr), pika::util::end(curr));
                    }
                }
            }

        protected:
            // Current element is a range (vector) of futures
            template <std::size_t I, typename Iter>
            void await_range(Iter&& next, Iter&& end)
            {
                using future_type =
                    typename std::iterator_traits<Iter>::value_type;

                pika::intrusive_ptr<when_each_frame> this_(this);
                for (/**/; next != end; ++next)
                {
                    auto next_future_data =
                        traits::detail::get_shared_state(*next);

                    if (next_future_data && !next_future_data->is_ready())
                    {
                        next_future_data->execute_deferred();

                        // execute_deferred might have made the future ready
                        if (!next_future_data->is_ready())
                        {
                            // Attach a continuation to this future which will
                            // re-evaluate it and continue to the next argument
                            // (if any).
                            next_future_data->set_on_completed(
                                [this_ = PIKA_MOVE(this_), next = PIKA_MOVE(next),
                                    end = PIKA_MOVE(end)]() mutable -> void {
                                    this_->template await_range<I>(
                                        PIKA_MOVE(next), PIKA_MOVE(end));
                                });

                            // explicitly destruct iterators as those might
                            // become dangling after we make ourselves ready
                            next = std::decay_t<Iter>{};
                            end = std::decay_t<Iter>{};
                            return;
                        }
                    }

                    // call supplied callback with or without index
                    if constexpr (pika::is_invocable_v<F, std::size_t,
                                      future_type>)
                    {
                        f_(count_, PIKA_MOVE(*next));
                    }
                    else
                    {
                        f_(PIKA_MOVE(*next));
                    }

                    if (++count_ == needed_count_)
                    {
                        this->set_value(util::unused);

                        // explicitly destruct iterators as those might
                        // become dangling after we make ourselves ready
                        next = std::decay_t<Iter>{};
                        end = std::decay_t<Iter>{};
                        return;
                    }
                }

                do_await<I + 1>();
            }

            // Current element is a simple future
            template <std::size_t I>
            PIKA_FORCEINLINE void await_future()
            {
                using future_type = pika::util::decay_unwrap_t<
                    typename pika::tuple_element<I, Tuple>::type>;

                pika::intrusive_ptr<when_each_frame> this_(this);

                future_type& fut = pika::get<I>(t_);
                auto next_future_data = traits::detail::get_shared_state(fut);
                if (next_future_data && !next_future_data->is_ready())
                {
                    next_future_data->execute_deferred();

                    // execute_deferred might have made the future ready
                    if (!next_future_data->is_ready())
                    {
                        // Attach a continuation to this future which will
                        // re-evaluate it and continue to the next argument
                        // (if any).
                        next_future_data->set_on_completed(
                            [this_ = PIKA_MOVE(this_)]() -> void {
                                this_->template await_future<I>();
                            });

                        return;
                    }
                }

                // call supplied callback with or without index
                if constexpr (pika::is_invocable_v<F, std::size_t, future_type>)
                {
                    f_(count_, PIKA_MOVE(fut));
                }
                else
                {
                    f_(PIKA_MOVE(fut));
                }

                if (++count_ == needed_count_)
                {
                    this->set_value(util::unused);
                    return;
                }

                do_await<I + 1>();
            }

        private:
            Tuple t_;
            F f_;
            std::size_t count_;
            std::size_t needed_count_;
        };
    }}    // namespace lcos::detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Future,
        typename Enable = std::enable_if_t<pika::traits::is_future_v<Future>>>
    pika::future<void> when_each(F&& func, std::vector<Future>& lazy_values)
    {
        using argument_type = pika::tuple<std::vector<Future>>;
        using frame_type =
            lcos::detail::when_each_frame<argument_type, std::decay_t<F>>;

        std::vector<Future> values;
        values.reserve(lazy_values.size());

        std::transform(lazy_values.begin(), lazy_values.end(),
            std::back_inserter(values), traits::acquire_future_disp());

        pika::intrusive_ptr<frame_type> p(
            new frame_type(pika::forward_as_tuple(PIKA_MOVE(values)),
                PIKA_FORWARD(F, func), values.size()));

        p->template do_await<0>();

        return pika::traits::future_access<typename frame_type::type>::create(
            PIKA_MOVE(p));
    }

    template <typename F, typename Future>
    pika::future<void>    //-V659
    when_each(F&& f, std::vector<Future>&& values)
    {
        return pika::when_each(PIKA_FORWARD(F, f), values);
    }

    template <typename F, typename Iterator,
        typename Enable =
            std::enable_if_t<pika::traits::is_iterator_v<Iterator>>>
    pika::future<Iterator> when_each(F&& f, Iterator begin, Iterator end)
    {
        using future_type =
            typename lcos::detail::future_iterator_traits<Iterator>::type;

        std::vector<future_type> values;
        traits::detail::reserve_if_random_access_by_range(values, begin, end);

        std::transform(begin, end, std::back_inserter(values),
            traits::acquire_future_disp());

        return pika::when_each(PIKA_FORWARD(F, f), values)
            .then(pika::launch::sync,
                [end = PIKA_MOVE(end)](pika::future<void> fut) -> Iterator {
                    fut.get();    // rethrow exceptions, if any
                    return end;
                });
    }

    template <typename F, typename Iterator,
        typename Enable =
            std::enable_if_t<pika::traits::is_iterator_v<Iterator>>>
    pika::future<Iterator> when_each_n(F&& f, Iterator begin, std::size_t count)
    {
        using future_type =
            typename lcos::detail::future_iterator_traits<Iterator>::type;

        std::vector<future_type> values;
        values.reserve(count);

        traits::acquire_future_disp func;
        while (count-- != 0)
        {
            values.push_back(func(*begin++));
        }

        return pika::when_each(PIKA_FORWARD(F, f), values)
            .then(pika::launch::sync,
                [begin = PIKA_MOVE(begin)](pika::future<void>&& fut) -> Iterator {
                    fut.get();    // rethrow exceptions, if any
                    return begin;
                });
    }

    template <typename F>
    inline pika::future<void> when_each(F&&)
    {
        return pika::make_ready_future();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    std::enable_if_t<!pika::traits::is_future_v<std::decay_t<F>> &&
            pika::util::all_of_v<pika::traits::is_future<Ts>...>,
        pika::future<void>>
    when_each(F&& f, Ts&&... ts)
    {
        using argument_type = pika::tuple<traits::acquire_future_t<Ts>...>;
        using frame_type =
            lcos::detail::when_each_frame<argument_type, std::decay_t<F>>;

        traits::acquire_future_disp func;
        argument_type values(func(PIKA_FORWARD(Ts, ts))...);

        pika::intrusive_ptr<frame_type> p(
            new frame_type(PIKA_MOVE(values), PIKA_FORWARD(F, f), sizeof...(Ts)));

        p->template do_await<0>();

        return pika::traits::future_access<typename frame_type::type>::create(
            PIKA_MOVE(p));
    }
}    // namespace pika

namespace pika::lcos {

    template <typename F, typename... Ts>
    PIKA_DEPRECATED_V(
        0, 1, "pika::lcos::when_each is deprecated. Use pika::when_each instead.")
    auto when_each(F&& f, Ts&&... ts)
    {
        return pika::when_each(PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
    }

    template <typename F, typename Iterator,
        typename Enable =
            std::enable_if_t<pika::traits::is_iterator_v<Iterator>>>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::when_each_n is deprecated. Use pika::when_each_n instead.")
    pika::future<Iterator> when_each_n(F&& f, Iterator begin, std::size_t count)
    {
        return pika::when_each_n(PIKA_FORWARD(F, f), begin, count);
    }
}    // namespace pika::lcos

#endif    // DOXYGEN
