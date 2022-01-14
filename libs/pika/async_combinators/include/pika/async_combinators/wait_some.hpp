//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file wait_some.hpp

#pragma once

#if defined(DOXYGEN)
namespace pika {
    /// The function \a wait_some is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing.
    ///
    /// \param n        [in] The number of futures out of the arguments which
    ///                 have to become ready in order for the function to return.
    /// \param first    [in] The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_all should wait.
    /// \param last     [in] The iterator pointing to the last element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_all should wait.
    ///
    /// \note The function \a wait_some returns after \a n futures have become
    ///       ready. All input futures are still valid after \a wait_some
    ///       returns.
    ///
    /// \note           The function wait_some will rethrow any exceptions
    ///                 captured by the futures while becoming ready. If this
    ///                 behavior is undesirable, use \a wait_some_nothrow
    ///                 instead.
    ///
    template <typename InputIter>
    void wait_some(std::size_t n, InputIter first, InputIter last);

    /// The function \a wait_some is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing.
    ///
    /// \param n        [in] The number of futures out of the arguments which
    ///                 have to become ready in order for the returned future
    ///                 to get ready.
    /// \param futures  [in] A vector holding an arbitrary amount of \a future
    ///                 or \a shared_future objects for which \a wait_some
    ///                 should wait.
    ///
    /// \note The function \a wait_some returns after \a n futures have become
    ///       ready. All input futures are still valid after \a wait_some
    ///       returns.
    ///
    /// \note           The function wait_some will rethrow any exceptions
    ///                 captured by the futures while becoming ready. If this
    ///                 behavior is undesirable, use \a wait_some_nothrow
    ///                 instead.
    ///
    template <typename R>
    void wait_some(std::size_t n, std::vector<future<R>>&& futures);

    /// The function \a wait_some is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing.
    ///
    /// \param n        [in] The number of futures out of the arguments which
    ///                 have to become ready in order for the returned future
    ///                 to get ready.
    /// \param futures  [in] An array holding an arbitrary amount of \a future
    ///                 or \a shared_future objects for which \a wait_some
    ///                 should wait.
    ///
    /// \note The function \a wait_some returns after \a n futures have become
    ///       ready. All input futures are still valid after \a wait_some
    ///       returns.
    ///
    /// \note           The function wait_some will rethrow any exceptions
    ///                 captured by the futures while becoming ready. If this
    ///                 behavior is undesirable, use \a wait_some_nothrow
    ///                 instead.
    ///
    template <typename R, std::size_t N>
    void wait_some(std::size_t n, std::array<future<R>, N>&& futures);

    /// The function \a wait_some is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing.
    ///
    /// \param n        [in] The number of futures out of the arguments which
    ///                 have to become ready in order for the returned future
    ///                 to get ready.
    /// \param futures  [in] An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a wait_some should wait.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a pika#throws the function
    ///                 will throw on error instead.
    ///
    /// \note The function \a wait_all returns after \a n futures have become
    ///       ready. All input futures are still valid after \a wait_some
    ///       returns.
    ///
    /// \note           The function wait_some will rethrow any exceptions
    ///                 captured by the futures while becoming ready. If this
    ///                 behavior is undesirable, use \a wait_some_nothrow
    ///                 instead.
    ///
    template <typename... T>
    void wait_some(std::size_t n, T&&... futures);

    /// The function \a wait_some_n is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing.
    ///
    /// \param n        [in] The number of futures out of the arguments which
    ///                 have to become ready in order for the returned future
    ///                 to get ready.
    /// \param first    [in] The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_all should wait.
    /// \param count    [in] The number of elements in the sequence starting at
    ///                 \a first.
    ///
    /// \note The function \a wait_some_n returns after \a n futures have become
    ///       ready. All input futures are still valid after \a wait_some_n
    ///       returns.
    ///
    /// \note           The function wait_some_n will rethrow any exceptions
    ///                 captured by the futures while becoming ready. If this
    ///                 behavior is undesirable, use \a wait_some_n_nothrow
    ///                 instead.
    ///
    template <typename InputIter>
    void wait_some_n(std::size_t n, InputIter first, std::size_t count);
}    // namespace pika
#else

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/async_combinators/detail/throw_if_exceptional.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/functional/deferred_call.hpp>
#include <pika/futures/future.hpp>
#include <pika/futures/traits/acquire_shared_state.hpp>
#include <pika/futures/traits/detail/future_traits.hpp>
#include <pika/futures/traits/future_access.hpp>
#include <pika/futures/traits/is_future.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/modules/errors.hpp>
#include <pika/preprocessor/strip_parens.hpp>
#include <pika/type_support/always_void.hpp>
#include <pika/type_support/pack.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace pika {
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename Sequence>
        struct wait_some;

        template <typename Sequence>
        struct set_wait_some_callback_impl
        {
            explicit set_wait_some_callback_impl(wait_some<Sequence>& wait)
              : wait_(wait)
            {
            }

            template <typename SharedState>
            void operator()(SharedState const& shared_state) const
            {
                if constexpr (!traits::is_shared_state_v<SharedState>)
                {
                    apply(shared_state);
                }
                else
                {
                    std::size_t counter =
                        wait_.count_.load(std::memory_order_acquire);

                    if (counter < wait_.needed_count_ && shared_state &&
                        !shared_state->is_ready())
                    {
                        // handle future only if not enough futures are ready yet
                        // also, do not touch any futures which are already ready
                        shared_state->execute_deferred();

                        // execute_deferred might have made the future ready
                        if (!shared_state->is_ready())
                        {
                            shared_state->set_on_completed(util::deferred_call(
                                &wait_some<Sequence>::on_future_ready,
                                wait_.shared_from_this(),
                                pika::execution_base::this_thread::agent()));
                            return;
                        }
                    }

                    if (wait_.count_.fetch_add(1) + 1 == wait_.needed_count_)
                    {
                        wait_.goal_reached_on_calling_thread_ = true;
                    }
                }
            }

            template <typename Tuple, std::size_t... Is>
            PIKA_FORCEINLINE void apply(
                Tuple const& tuple, pika::util::index_pack<Is...>) const
            {
                int const _sequencer[] = {
                    0, (((*this)(pika::get<Is>(tuple))), 0)...};
                (void) _sequencer;
            }

            template <typename... Ts>
            PIKA_FORCEINLINE void apply(pika::tuple<Ts...> const& sequence) const
            {
                apply(sequence, pika::util::make_index_pack_t<sizeof...(Ts)>());
            }

            template <typename Sequence_>
            PIKA_FORCEINLINE void apply(Sequence_ const& sequence) const
            {
                std::for_each(sequence.begin(), sequence.end(), *this);
            }

            wait_some<Sequence>& wait_;
        };

        template <typename Sequence>
        void set_on_completed_callback(wait_some<Sequence>& wait)
        {
            set_wait_some_callback_impl<Sequence> callback(wait);
            callback.apply(wait.values_);
        }

        template <typename Sequence>
        struct wait_some
          : std::enable_shared_from_this<wait_some<Sequence>>    //-V690
        {
        public:
            void on_future_ready(pika::execution_base::agent_ref ctx)
            {
                if (count_.fetch_add(1) + 1 == needed_count_)
                {
                    // reactivate waiting thread only if it's not us
                    if (ctx != pika::execution_base::this_thread::agent())
                    {
                        ctx.resume();
                    }
                    else
                    {
                        goal_reached_on_calling_thread_ = true;
                    }
                }
            }

        private:
            wait_some(wait_some const&) = delete;
            wait_some(wait_some&&) = delete;

            wait_some& operator=(wait_some const&) = delete;
            wait_some& operator=(wait_some&&) = delete;

        public:
            using argument_type = Sequence;

            wait_some(argument_type const& values, std::size_t n)
              : values_(values)
              , count_(0)
              , needed_count_(n)
              , goal_reached_on_calling_thread_(false)
            {
            }

            void operator()()
            {
                // set callback functions to executed wait future is ready
                set_on_completed_callback(*this);

                // if all of the requested futures are already set, our
                // callback above has already been called often enough, otherwise
                // we suspend ourselves
                if (!goal_reached_on_calling_thread_)
                {
                    // wait for any of the futures to return to become ready
                    pika::execution_base::this_thread::suspend(
                        "pika::detail::wait_some::operator()");
                }

                // at least N futures should be ready
                PIKA_ASSERT(
                    count_.load(std::memory_order_acquire) >= needed_count_);
            }

            argument_type const& values_;
            std::atomic<std::size_t> count_;
            std::size_t const needed_count_;
            bool goal_reached_on_calling_thread_;
        };

        template <typename T>
        auto get_wait_some_frame(T const& values, std::size_t n)
        {
            return std::make_shared<pika::detail::wait_some<T>>(values, n);
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    void wait_some_nothrow(std::size_t n, std::vector<Future> const& values)
    {
        static_assert(
            pika::traits::is_future_v<Future>, "invalid use of pika::wait_some");

        if (n == 0)
        {
            return;
        }

        if (n > values.size())
        {
            PIKA_THROW_EXCEPTION(pika::bad_parameter, "pika::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }

        auto lazy_values = traits::acquire_shared_state_disp()(values);
        auto f = detail::get_wait_some_frame(lazy_values, n);
        (*f)();
    }

    template <typename Future>
    void wait_some(std::size_t n, std::vector<Future> const& values)
    {
        pika::wait_some_nothrow(n, values);
        pika::detail::throw_if_exceptional(values);
    }

    template <typename Future>
    void wait_some_nothrow(std::size_t n, std::vector<Future>& values)
    {
        return pika::wait_some_nothrow(
            n, const_cast<std::vector<Future> const&>(values));
    }

    template <typename Future>
    void wait_some(std::size_t n, std::vector<Future>& values)
    {
        pika::wait_some_nothrow(
            n, const_cast<std::vector<Future> const&>(values));
        pika::detail::throw_if_exceptional(values);
    }

    template <typename Future>
    void wait_some_nothrow(std::size_t n, std::vector<Future>&& values)
    {
        return pika::wait_some_nothrow(
            n, const_cast<std::vector<Future> const&>(values));
    }

    template <typename Future>
    void wait_some(std::size_t n, std::vector<Future>&& values)
    {
        pika::wait_some_nothrow(
            n, const_cast<std::vector<Future> const&>(values));
        pika::detail::throw_if_exceptional(values);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, std::size_t N>
    void wait_some_nothrow(std::size_t n, std::array<Future, N> const& values)
    {
        static_assert(
            pika::traits::is_future_v<Future>, "invalid use of wait_some");

        if (n == 0)
        {
            return;
        }

        if (n > values.size())
        {
            PIKA_THROW_EXCEPTION(pika::bad_parameter, "pika::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }

        auto lazy_values = traits::acquire_shared_state_disp()(values);
        auto f = detail::get_wait_some_frame(lazy_values, n);
        (*f)();
    }

    template <typename Future, std::size_t N>
    void wait_some(std::size_t n, std::array<Future, N> const& lazy_values)
    {
        pika::wait_some_nothrow(n, lazy_values);
        pika::detail::throw_if_exceptional(lazy_values);
    }

    template <typename Future, std::size_t N>
    void wait_some_nothrow(std::size_t n, std::array<Future, N>& lazy_values)
    {
        pika::wait_some_nothrow(
            n, const_cast<std::array<Future, N> const&>(lazy_values));
    }

    template <typename Future, std::size_t N>
    void wait_some(std::size_t n, std::array<Future, N>& lazy_values)
    {
        pika::wait_some_nothrow(
            n, const_cast<std::array<Future, N> const&>(lazy_values));
        pika::detail::throw_if_exceptional(lazy_values);
    }

    template <typename Future, std::size_t N>
    void wait_some_nothrow(std::size_t n, std::array<Future, N>&& lazy_values)
    {
        pika::wait_some_nothrow(
            n, const_cast<std::array<Future, N> const&>(lazy_values));
    }

    template <typename Future, std::size_t N>
    void wait_some(std::size_t n, std::array<Future, N>&& lazy_values)
    {
        pika::wait_some_nothrow(
            n, const_cast<std::array<Future, N> const&>(lazy_values));
        pika::detail::throw_if_exceptional(lazy_values);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator,
        typename Enable =
            std::enable_if_t<pika::traits::is_iterator_v<Iterator>>>
    void wait_some_nothrow(std::size_t n, Iterator begin, Iterator end)
    {
        auto values = traits::acquire_shared_state<Iterator>()(begin, end);
        auto f = detail::get_wait_some_frame(values, n);
        (*f)();
    }

    template <typename Iterator,
        typename Enable =
            std::enable_if_t<pika::traits::is_iterator_v<Iterator>>>
    void wait_some(std::size_t n, Iterator begin, Iterator end)
    {
        auto values = traits::acquire_shared_state<Iterator>()(begin, end);
        auto f = detail::get_wait_some_frame(values, n);
        (*f)();
        pika::detail::throw_if_exceptional(values);
    }

    template <typename Iterator,
        typename Enable =
            std::enable_if_t<pika::traits::is_iterator_v<Iterator>>>
    void wait_some_n_nothrow(std::size_t n, Iterator begin, std::size_t count)
    {
        auto values = traits::acquire_shared_state<Iterator>()(begin, count);
        auto f = detail::get_wait_some_frame(values, n);
        (*f)();
    }

    template <typename Iterator,
        typename Enable =
            std::enable_if_t<pika::traits::is_iterator_v<Iterator>>>
    void wait_some_n(std::size_t n, Iterator begin, std::size_t count)
    {
        auto values = traits::acquire_shared_state<Iterator>()(begin, count);
        auto f = detail::get_wait_some_frame(values, n);
        (*f)();
        pika::detail::throw_if_exceptional(values);
    }

    inline void wait_some_nothrow(std::size_t n)
    {
        if (n != 0)
        {
            PIKA_THROW_EXCEPTION(pika::bad_parameter, "pika::wait_some",
                "number of results to wait for is out of bounds");
        }
    }

    inline void wait_some(std::size_t n)
    {
        wait_some_nothrow(n);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    void wait_some_nothrow(std::size_t n, pika::future<T>&& f)
    {
        if (n != 1)
        {
            PIKA_THROW_EXCEPTION(pika::bad_parameter, "pika::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }

        f.wait();
    }

    template <typename T>
    void wait_some(std::size_t n, pika::future<T>&& f)
    {
        pika::wait_some_nothrow(n, PIKA_MOVE(f));
        pika::detail::throw_if_exceptional(f);
    }

    template <typename T>
    void wait_some_nothrow(std::size_t n, pika::shared_future<T>&& f)
    {
        if (n != 1)
        {
            PIKA_THROW_EXCEPTION(pika::bad_parameter, "pika::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }

        f.wait();
    }

    template <typename T>
    void wait_some(std::size_t n, pika::shared_future<T>&& f)
    {
        pika::wait_some_nothrow(n, PIKA_MOVE(f));
        pika::detail::throw_if_exceptional(f);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Ts>
    void wait_some_nothrow(std::size_t n, Ts&&... ts)
    {
        if (n == 0)
        {
            return;
        }

        if (n > sizeof...(Ts))
        {
            PIKA_THROW_EXCEPTION(pika::bad_parameter, "pika::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }

        using result_type =
            pika::tuple<traits::detail::shared_state_ptr_for_t<Ts>...>;

        result_type values(traits::detail::get_shared_state(ts)...);
        auto f = detail::get_wait_some_frame(values, n);
        (*f)();
    }

    template <typename... Ts>
    void wait_some(std::size_t n, Ts&&... ts)
    {
        pika::wait_some_nothrow(n, ts...);
        pika::detail::throw_if_exceptional(PIKA_FORWARD(Ts, ts)...);
    }
}    // namespace pika

namespace pika::lcos {

    template <typename Future>
    PIKA_DEPRECATED_V(
        0, 1, "pika::lcos::wait_some is deprecated. Use pika::wait_some instead.")
    void wait_some(std::size_t n, std::vector<Future> const& lazy_values,
        error_code& = throws)
    {
        pika::wait_some(n, lazy_values);
    }

    template <typename Future>
    PIKA_DEPRECATED_V(
        0, 1, "pika::lcos::wait_some is deprecated. Use pika::wait_some instead.")
    void wait_some(
        std::size_t n, std::vector<Future>& lazy_values, error_code& = throws)
    {
        pika::wait_some(n, const_cast<std::vector<Future> const&>(lazy_values));
    }

    template <typename Future>
    PIKA_DEPRECATED_V(
        0, 1, "pika::lcos::wait_some is deprecated. Use pika::wait_some instead.")
    void wait_some(
        std::size_t n, std::vector<Future>&& lazy_values, error_code& = throws)
    {
        pika::wait_some(n, const_cast<std::vector<Future> const&>(lazy_values));
    }

    template <typename Future, std::size_t N>
    PIKA_DEPRECATED_V(
        0, 1, "pika::lcos::wait_some is deprecated. Use pika::wait_some instead.")
    void wait_some(std::size_t n, std::array<Future, N> const& lazy_values,
        error_code& = throws)
    {
        pika::wait_some(n, lazy_values);
    }

    template <typename Future, std::size_t N>
    PIKA_DEPRECATED_V(
        0, 1, "pika::lcos::wait_some is deprecated. Use pika::wait_some instead.")
    void wait_some(
        std::size_t n, std::array<Future, N>& lazy_values, error_code& = throws)
    {
        pika::wait_some(
            n, const_cast<std::array<Future, N> const&>(lazy_values));
    }

    template <typename Future, std::size_t N>
    PIKA_DEPRECATED_V(
        0, 1, "pika::lcos::wait_some is deprecated. Use pika::wait_some instead.")
    void wait_some(std::size_t n, std::array<Future, N>&& lazy_values,
        error_code& = throws)
    {
        pika::wait_some(
            n, const_cast<std::array<Future, N> const&>(lazy_values));
    }

    template <typename Iterator,
        typename Enable =
            std::enable_if_t<pika::traits::is_iterator_v<Iterator>>>
    PIKA_DEPRECATED_V(
        0, 1, "pika::lcos::wait_some is deprecated. Use pika::wait_some instead.")
    void wait_some(
        std::size_t n, Iterator begin, Iterator end, error_code& = throws)
    {
        pika::wait_some(n, begin, end);
    }

    template <typename Iterator,
        typename Enable =
            std::enable_if_t<pika::traits::is_iterator_v<Iterator>>>
    PIKA_DEPRECATED_V(
        0, 1, "pika::lcos::wait_some is deprecated. Use pika::wait_some instead.")
    Iterator wait_some_n(
        std::size_t n, Iterator begin, std::size_t count, error_code& = throws)
    {
        pika::wait_some(n, begin, count);
    }

    PIKA_DEPRECATED_V(
        0, 1, "pika::lcos::wait_some is deprecated. Use pika::wait_some instead.")
    inline void wait_some(std::size_t n, error_code& = throws)
    {
        pika::wait_some(n);
    }

    template <typename T>
    PIKA_DEPRECATED_V(
        0, 1, "pika::lcos::wait_some is deprecated. Use pika::wait_some instead.")
    void wait_some(std::size_t n, pika::future<T>&& f, error_code& = throws)
    {
        pika::wait_some(n, PIKA_MOVE(f));
    }

    template <typename T>
    PIKA_DEPRECATED_V(
        0, 1, "pika::lcos::wait_some is deprecated. Use pika::wait_some instead.")
    void wait_some(
        std::size_t n, pika::shared_future<T>&& f, error_code& = throws)
    {
        pika::wait_some(n, PIKA_MOVE(f));
    }

    template <typename... Ts>
    PIKA_DEPRECATED_V(
        0, 1, "pika::lcos::wait_some is deprecated. Use pika::wait_some instead.")
    void wait_some(std::size_t n, error_code&, Ts&&... ts)
    {
        pika::wait_some(n, PIKA_FORWARD(Ts, ts)...);
    }

    template <typename... Ts>
    PIKA_DEPRECATED_V(
        0, 1, "pika::lcos::wait_some is deprecated. Use pika::wait_some instead.")
    void wait_some(std::size_t n, Ts&&... ts)
    {
        pika::wait_some(n, PIKA_FORWARD(Ts, ts)...);
    }
}    // namespace pika::lcos

#endif    // DOXYGEN
