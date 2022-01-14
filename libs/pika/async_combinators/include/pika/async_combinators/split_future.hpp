//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file lcos/split_future.hpp

#pragma once

#if defined(DOXYGEN)
namespace pika {
    /// The function \a split_future is an operator allowing to split a given
    /// future of a sequence of values (any tuple, std::pair, or std::array)
    /// into an equivalent container of futures where each future represents
    /// one of the values from the original future. In some sense this function
    /// provides the inverse operation of \a when_all.
    ///
    /// \param f    [in] A future holding an arbitrary sequence of values stored
    ///             in a tuple-like container. This facility supports
    ///             \a pika::tuple<>, \a std::pair<T1, T2>, and
    ///             \a std::array<T, N>
    ///
    /// \return     Returns an equivalent container (same container type as
    ///             passed as the argument) of futures, where each future refers
    ///             to the corresponding value in the input parameter. All of
    ///             the returned futures become ready once the input future has
    ///             become ready. If the input future is exceptional, all output
    ///             futures will be exceptional as well.
    ///
    /// \note       The following cases are special:
    /// \code
    ///     tuple<future<void> > split_future(future<tuple<> > && f);
    ///     array<future<void>, 1> split_future(future<array<T, 0> > && f);
    /// \endcode
    ///             here the returned futures are directly representing the
    ///             futures which were passed to the function.
    ///
    template <typename... Ts>
    inline tuple<future<Ts>...> split_future(future<tuple<Ts...>>&& f);

    /// The function \a split_future is an operator allowing to split a given
    /// future of a sequence of values (any std::vector)
    /// into a std::vector of futures where each future represents
    /// one of the values from the original std::vector. In some sense this
    /// function provides the inverse operation of \a when_all.
    ///
    /// \param f    [in] A future holding an arbitrary sequence of values stored
    ///             in a std::vector.
    /// \param size [in] The number of elements the vector will hold once the
    ///             input future has become ready
    ///
    /// \return     Returns a std::vector of futures, where each future refers
    ///             to the corresponding value in the input parameter. All of
    ///             the returned futures become ready once the input future has
    ///             become ready. If the input future is exceptional, all output
    ///             futures will be exceptional as well.
    ///
    template <typename T>
    inline std::vector<future<T>> split_future(
        future<std::vector<T>>&& f, std::size_t size);
}    // namespace pika

#else    // DOXYGEN

#include <pika/local/config.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/errors/try_catch_exception_ptr.hpp>
#include <pika/functional/deferred_call.hpp>
#include <pika/futures/detail/future_data.hpp>
#include <pika/futures/future.hpp>
#include <pika/futures/packaged_continuation.hpp>
#include <pika/futures/traits/acquire_future.hpp>
#include <pika/futures/traits/acquire_shared_state.hpp>
#include <pika/futures/traits/future_access.hpp>
#include <pika/futures/traits/future_traits.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/memory.hpp>
#include <pika/type_support/pack.hpp>
#include <pika/type_support/unused.hpp>

#include <array>
#include <cstddef>
#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace lcos {
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename ContResult>
        class split_nth_continuation : public future_data<ContResult>
        {
            typedef future_data<ContResult> base_type;

        private:
            template <std::size_t I, typename T>
            void on_ready(
                typename traits::detail::shared_state_ptr_for<T>::type const&
                    state)
            {
                pika::detail::try_catch_exception_ptr(
                    [&]() {
                        typedef
                            typename traits::future_traits<T>::type result_type;
                        result_type* result = state->get_result();
                        this->base_type::set_value(
                            PIKA_MOVE(pika::get<I>(*result)));
                    },
                    [&](std::exception_ptr ep) {
                        this->base_type::set_exception(PIKA_MOVE(ep));
                    });
            }

        public:
            template <std::size_t I, typename Future>
            void attach(Future& future)
            {
                typedef
                    typename traits::detail::shared_state_ptr_for<Future>::type
                        shared_state_ptr;

                // Bind an on_completed handler to this future which will wait
                // for the future and will transfer its result to the new
                // future.
                pika::intrusive_ptr<split_nth_continuation> this_(this);
                shared_state_ptr const& state =
                    pika::traits::detail::get_shared_state(future);

                state->execute_deferred();
                state->set_on_completed(util::deferred_call(
                    &split_nth_continuation::on_ready<I, Future>,
                    PIKA_MOVE(this_), state));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Result, typename Tuple, std::size_t I,
            typename Future>
        inline typename pika::traits::detail::shared_state_ptr<
            typename pika::tuple_element<I, Tuple>::type>::type
        extract_nth_continuation(Future& future)
        {
            typedef split_nth_continuation<Result> shared_state;

            typename pika::traits::detail::shared_state_ptr<Result>::type p(
                new shared_state());

            static_cast<shared_state*>(p.get())->template attach<I>(future);
            return p;
        }

        ///////////////////////////////////////////////////////////////////////
        template <std::size_t I, typename Tuple>
        PIKA_FORCEINLINE pika::future<typename pika::tuple_element<I, Tuple>::type>
        extract_nth_future(pika::future<Tuple>& future)
        {
            typedef typename pika::tuple_element<I, Tuple>::type result_type;

            return pika::traits::future_access<pika::future<result_type>>::create(
                extract_nth_continuation<result_type, Tuple, I>(future));
        }

        template <std::size_t I, typename Tuple>
        PIKA_FORCEINLINE pika::future<typename pika::tuple_element<I, Tuple>::type>
        extract_nth_future(pika::shared_future<Tuple>& future)
        {
            typedef typename pika::tuple_element<I, Tuple>::type result_type;

            return pika::traits::future_access<pika::future<result_type>>::create(
                extract_nth_continuation<result_type, Tuple, I>(future));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename... Ts, std::size_t... Is>
        PIKA_FORCEINLINE pika::tuple<pika::future<Ts>...> split_future_helper(
            pika::future<pika::tuple<Ts...>>&& f, pika::util::index_pack<Is...>)
        {
            return pika::make_tuple(extract_nth_future<Is>(f)...);
        }

        template <typename... Ts, std::size_t... Is>
        PIKA_FORCEINLINE pika::tuple<pika::future<Ts>...> split_future_helper(
            pika::shared_future<pika::tuple<Ts...>>&& f,
            pika::util::index_pack<Is...>)
        {
            return pika::make_tuple(extract_nth_future<Is>(f)...);
        }

        ///////////////////////////////////////////////////////////////////////
#if defined(PIKA_DATASTRUCTURES_HAVE_ADAPT_STD_TUPLE)
        template <typename... Ts, std::size_t... Is>
        PIKA_FORCEINLINE std::tuple<pika::future<Ts>...> split_future_helper(
            pika::future<std::tuple<Ts...>>&& f, pika::util::index_pack<Is...>)
        {
            return std::make_tuple(extract_nth_future<Is>(f)...);
        }

        template <typename... Ts, std::size_t... Is>
        PIKA_FORCEINLINE std::tuple<pika::future<Ts>...> split_future_helper(
            pika::shared_future<std::tuple<Ts...>>&& f,
            pika::util::index_pack<Is...>)
        {
            return std::make_tuple(extract_nth_future<Is>(f)...);
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        template <typename T1, typename T2>
        PIKA_FORCEINLINE std::pair<pika::future<T1>, pika::future<T2>>
        split_future_helper(pika::future<std::pair<T1, T2>>&& f)
        {
            return std::make_pair(
                extract_nth_future<0>(f), extract_nth_future<1>(f));
        }

        template <typename T1, typename T2>
        PIKA_FORCEINLINE std::pair<pika::future<T1>, pika::future<T2>>
        split_future_helper(pika::shared_future<std::pair<T1, T2>>&& f)
        {
            return std::make_pair(
                extract_nth_future<0>(f), extract_nth_future<1>(f));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ContResult>
        class split_continuation : public future_data<ContResult>
        {
            typedef future_data<ContResult> base_type;

        private:
            template <typename T>
            void on_ready(std::size_t i,
                typename traits::detail::shared_state_ptr_for<T>::type const&
                    state)
            {
                pika::detail::try_catch_exception_ptr(
                    [&]() {
                        typedef
                            typename traits::future_traits<T>::type result_type;
                        result_type* result = state->get_result();
                        if (i >= result->size())
                        {
                            PIKA_THROW_EXCEPTION(length_error,
                                "split_continuation::on_ready",
                                "index out of bounds");
                        }
                        this->base_type::set_value(PIKA_MOVE((*result)[i]));
                    },
                    [&](std::exception_ptr ep) {
                        this->base_type::set_exception(PIKA_MOVE(ep));
                    });
            }

        public:
            template <typename Future>
            void attach(std::size_t i, Future& future)
            {
                typedef
                    typename traits::detail::shared_state_ptr_for<Future>::type
                        shared_state_ptr;

                // Bind an on_completed handler to this future which will wait
                // for the future and will transfer its result to the new
                // future.
                pika::intrusive_ptr<split_continuation> this_(this);
                shared_state_ptr const& state =
                    pika::traits::detail::get_shared_state(future);

                state->execute_deferred();
                state->set_on_completed(
                    util::deferred_call(&split_continuation::on_ready<Future>,
                        PIKA_MOVE(this_), i, state));
            }
        };

        template <typename T, typename Future>
        inline pika::future<T> extract_future_array(
            std::size_t i, Future& future)
        {
            typedef split_continuation<T> shared_state;

            typename pika::traits::detail::shared_state_ptr<T>::type p(
                new shared_state());

            static_cast<shared_state*>(p.get())->attach(i, future);
            return pika::traits::future_access<pika::future<T>>::create(p);
        }

        template <std::size_t N, typename T, typename Future>
        inline std::array<pika::future<T>, N> split_future_helper_array(
            Future&& f)
        {
            std::array<pika::future<T>, N> result;

            for (std::size_t i = 0; i != N; ++i)
                result[i] = extract_future_array<T>(i, f);

            return result;
        }

        template <typename T, typename Future>
        inline std::vector<pika::future<T>> split_future_helper_vector(
            Future&& f, std::size_t size)
        {
            std::vector<pika::future<T>> result;
            result.reserve(size);

            for (std::size_t i = 0; i != size; ++i)
                result.push_back(extract_future_array<T>(i, f));

            return result;
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Ts>
    PIKA_FORCEINLINE pika::tuple<pika::future<Ts>...> split_future(
        pika::future<pika::tuple<Ts...>>&& f)
    {
        return detail::split_future_helper(PIKA_MOVE(f),
            typename pika::util::make_index_pack<sizeof...(Ts)>::type());
    }

    PIKA_FORCEINLINE pika::tuple<pika::future<void>> split_future(
        pika::future<pika::tuple<>>&& f)
    {
        return pika::make_tuple(pika::future<void>(PIKA_MOVE(f)));
    }

    template <typename... Ts>
    PIKA_FORCEINLINE pika::tuple<pika::future<Ts>...> split_future(
        pika::shared_future<pika::tuple<Ts...>>&& f)
    {
        return detail::split_future_helper(PIKA_MOVE(f),
            typename pika::util::make_index_pack<sizeof...(Ts)>::type());
    }

    PIKA_FORCEINLINE pika::tuple<pika::future<void>> split_future(
        pika::shared_future<pika::tuple<>>&& f)
    {
        return pika::make_tuple(pika::make_future<void>(PIKA_MOVE(f)));
    }

    ///////////////////////////////////////////////////////////////////////////
#if defined(PIKA_DATASTRUCTURES_HAVE_ADAPT_STD_TUPLE)
    template <typename... Ts>
    PIKA_FORCEINLINE std::tuple<pika::future<Ts>...> split_future(
        pika::future<std::tuple<Ts...>>&& f)
    {
        return detail::split_future_helper(PIKA_MOVE(f),
            typename pika::util::make_index_pack<sizeof...(Ts)>::type());
    }

    PIKA_FORCEINLINE std::tuple<pika::future<void>> split_future(
        pika::future<std::tuple<>>&& f)
    {
        return std::make_tuple(pika::future<void>(PIKA_MOVE(f)));
    }

    template <typename... Ts>
    PIKA_FORCEINLINE std::tuple<pika::future<Ts>...> split_future(
        pika::shared_future<std::tuple<Ts...>>&& f)
    {
        return detail::split_future_helper(PIKA_MOVE(f),
            typename pika::util::make_index_pack<sizeof...(Ts)>::type());
    }

    PIKA_FORCEINLINE std::tuple<pika::future<void>> split_future(
        pika::shared_future<std::tuple<>>&& f)
    {
        return std::make_tuple(pika::make_future<void>(PIKA_MOVE(f)));
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename T1, typename T2>
    PIKA_FORCEINLINE std::pair<pika::future<T1>, pika::future<T2>> split_future(
        pika::future<std::pair<T1, T2>>&& f)
    {
        return detail::split_future_helper(PIKA_MOVE(f));
    }

    template <typename T1, typename T2>
    PIKA_FORCEINLINE std::pair<pika::future<T1>, pika::future<T2>> split_future(
        pika::shared_future<std::pair<T1, T2>>&& f)
    {
        return detail::split_future_helper(PIKA_MOVE(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <std::size_t N, typename T>
    PIKA_FORCEINLINE std::array<pika::future<T>, N> split_future(
        pika::future<std::array<T, N>>&& f)
    {
        return detail::split_future_helper_array<N, T>(PIKA_MOVE(f));
    }

    template <typename T>
    PIKA_FORCEINLINE std::array<pika::future<void>, 1> split_future(
        pika::future<std::array<T, 0>>&& f)
    {
        std::array<pika::future<void>, 1> result;
        result[0] = pika::future<void>(PIKA_MOVE(f));
        return result;
    }

    template <std::size_t N, typename T>
    PIKA_FORCEINLINE std::array<pika::future<T>, N> split_future(
        pika::shared_future<std::array<T, N>>&& f)
    {
        return detail::split_future_helper_array<N, T>(PIKA_MOVE(f));
    }

    template <typename T>
    PIKA_FORCEINLINE std::array<pika::future<void>, 1> split_future(
        pika::shared_future<std::array<T, 0>>&& f)
    {
        std::array<pika::future<void>, 1> result;
        result[0] = pika::make_future<void>(PIKA_MOVE(f));
        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    PIKA_FORCEINLINE std::vector<pika::future<T>> split_future(
        pika::future<std::vector<T>>&& f, std::size_t size)
    {
        return detail::split_future_helper_vector<T>(PIKA_MOVE(f), size);
    }

    template <typename T>
    PIKA_FORCEINLINE std::vector<pika::future<T>> split_future(
        pika::shared_future<std::vector<T>>&& f, std::size_t size)
    {
        return detail::split_future_helper_vector<T>(PIKA_MOVE(f), size);
    }
}}    // namespace pika::lcos

namespace pika {
    using lcos::split_future;
}
#endif
