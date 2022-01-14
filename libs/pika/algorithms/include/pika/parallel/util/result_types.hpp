//  Copyright (c) 2020-2021 Hartmut Kaiser
//  Copyright (c) 2021 Giannis Gonidelis
//  Copyright (c) 2021 Chuanqiu He
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/copy.hpp

// make sure inspect doesn't get upset about the unproteced min()/max() below
// pikainspect:nominmax

#pragma once

#include <pika/local/config.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/modules/futures.hpp>

#include <type_traits>
#include <utility>

namespace pika { namespace parallel { namespace util {

    ///////////////////////////////////////////////////////////////////////////
    template <typename I1, typename I2>
    struct in_in_result
    {
        PIKA_NO_UNIQUE_ADDRESS I1 in1;
        PIKA_NO_UNIQUE_ADDRESS I2 in2;

        template <typename II1, typename II2,
            typename Enable =
                std::enable_if_t<std::is_convertible_v<I1 const&, II1> &&
                    std::is_convertible_v<I2 const&, II2>>>
        constexpr operator in_in_result<II1, II2>() const&
        {
            return {in1, in2};
        }

        template <typename II1, typename II2,
            typename Enable = std::enable_if_t<std::is_convertible_v<I1, II1> &&
                std::is_convertible_v<I2, II2>>>
        constexpr operator in_in_result<II1, II2>() &&
        {
            return {PIKA_MOVE(in1), PIKA_MOVE(in2)};
        }

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            // clang-format off
            ar & in1 & in2;
            // clang-format on
        }
    };

    template <typename I1, typename I2>
    I2 get_in2_element(util::in_in_result<I1, I2>&& p)
    {
        return p.in2;
    }

    template <typename I1, typename I2>
    pika::future<I2> get_in2_element(pika::future<util::in_in_result<I1, I2>>&& f)
    {
        return pika::make_future<I2>(
            PIKA_MOVE(f), [](util::in_in_result<I1, I2>&& p) { return p.in2; });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename I, typename O>
    struct in_out_result
    {
        PIKA_NO_UNIQUE_ADDRESS I in;
        PIKA_NO_UNIQUE_ADDRESS O out;

        template <typename I2, typename O2,
            typename Enable =
                std::enable_if_t<std::is_convertible_v<I const&, I2> &&
                    std::is_convertible_v<O const&, O2>>>
        constexpr operator in_out_result<I2, O2>() const&
        {
            return {in, out};
        }

        template <typename I2, typename O2,
            typename Enable = std::enable_if_t<std::is_convertible_v<I, I2> &&
                std::is_convertible_v<O, O2>>>
        constexpr operator in_out_result<I2, O2>() &&
        {
            return {PIKA_MOVE(in), PIKA_MOVE(out)};
        }

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            // clang-format off
            ar & in & out;
            // clang-format on
        }
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename I, typename O>
    std::pair<I, O> get_pair(util::in_out_result<I, O>&& p)
    {
        return std::pair<I, O>{p.in, p.out};
    }

    template <typename I, typename O>
    O get_second_element(util::in_out_result<I, O>&& p)
    {
        return p.out;
    }

    template <typename I, typename O>
    pika::future<std::pair<I, O>> get_pair(
        pika::future<util::in_out_result<I, O>>&& f)
    {
        return pika::make_future<std::pair<I, O>>(
            PIKA_MOVE(f), [](util::in_out_result<I, O>&& p) {
                return std::pair<I, O>{p.in, p.out};
            });
    }

    template <typename I, typename O>
    pika::future<O> get_second_element(
        pika::future<util::in_out_result<I, O>>&& f)
    {
        return pika::make_future<O>(
            PIKA_MOVE(f), [](util::in_out_result<I, O>&& p) { return p.out; });
    }

    // converst a in_out_result into a iterator_range
    template <typename I, typename O>
    pika::util::iterator_range<I, O> get_subrange(in_out_result<I, O> const& ior)
    {
        return pika::util::iterator_range<I, O>(ior.in, ior.out);
    }

    template <typename I, typename O>
    pika::future<pika::util::iterator_range<I, O>> get_subrange(
        pika::future<in_out_result<I, O>>&& ior)
    {
        return pika::make_future<pika::util::iterator_range<I, O>>(
            PIKA_MOVE(ior), [](in_out_result<I, O>&& ior) {
                return pika::util::iterator_range<I, O>(ior.in, ior.out);
            });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct min_max_result
    {
        PIKA_NO_UNIQUE_ADDRESS T min;
        PIKA_NO_UNIQUE_ADDRESS T max;

        template <typename T2,
            typename Enable =
                std::enable_if_t<std::is_convertible_v<T const&, T>>>
        constexpr operator min_max_result<T2>() const&
        {
            return {min, max};
        }

        template <typename T2,
            typename Enable = std::enable_if_t<std::is_convertible_v<T, T2>>>
        constexpr operator min_max_result<T2>() &&
        {
            return {PIKA_MOVE(min), PIKA_MOVE(max)};
        }

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            // clang-format off
            ar & min & max;
            // clang-format on
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename I1, typename I2, typename O>
    struct in_in_out_result
    {
        PIKA_NO_UNIQUE_ADDRESS I1 in1;
        PIKA_NO_UNIQUE_ADDRESS I2 in2;
        PIKA_NO_UNIQUE_ADDRESS O out;

        template <typename II1, typename II2, typename O1,
            typename Enable = typename std::enable_if_t<
                std::is_convertible_v<I1 const&, II1> &&
                std::is_convertible_v<I2 const&, II2> &&
                std::is_convertible_v<O const&, O1>>>
        constexpr operator in_in_out_result<II1, II2, O1>() const&
        {
            return {in1, in2, out};
        }

        template <typename II2, typename II1, typename O1,
            typename Enable = typename std::enable_if_t<
                std::is_convertible_v<I1, II1> &&
                std::is_convertible_v<I2, II2> && std::is_convertible_v<O, O1>>>
        constexpr operator in_in_out_result<II1, II2, O1>() &&
        {
            return {PIKA_MOVE(in1), PIKA_MOVE(in2), PIKA_MOVE(out)};
        }

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            // clang-format off
            ar & in1 & in2 & out;
            // clang-format on
        }
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename I1, typename I2, typename O>
    O get_third_element(util::in_in_out_result<I1, I2, O>&& p)
    {
        return p.out;
    }

    template <typename I1, typename I2, typename O>
    pika::future<O> get_third_element(
        pika::future<util::in_in_out_result<I1, I2, O>>&& f)
    {
        return pika::make_future<O>(
            PIKA_MOVE(f), [](in_in_out_result<I1, I2, O>&& p) { return p.out; });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename I, typename O1, typename O2>
    struct in_out_out_result
    {
        PIKA_NO_UNIQUE_ADDRESS I in;
        PIKA_NO_UNIQUE_ADDRESS O1 out1;
        PIKA_NO_UNIQUE_ADDRESS O2 out2;

        template <typename II, typename OO1, typename OO2,
            typename Enable =
                typename std::enable_if_t<std::is_convertible_v<I const&, II> &&
                    std::is_convertible_v<O1 const&, OO1> &&
                    std::is_convertible_v<O2 const&, OO2>>>
        constexpr operator in_out_out_result<II, OO1, OO2>() const&
        {
            return {in, out1, out2};
        }

        template <typename II, typename OO1, typename OO2,
            typename Enable =
                typename std::enable_if_t<std::is_convertible_v<I, II> &&
                    std::is_convertible_v<O1, OO1> &&
                    std::is_convertible_v<O2, OO2>>>
        constexpr operator in_out_out_result<II, OO1, OO2>() &&
        {
            return {PIKA_MOVE(in), PIKA_MOVE(out1), PIKA_MOVE(out2)};
        }

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            // clang-format off
            ar & in & out1 & out2;
            // clang-format on
        }
    };

    template <typename... Ts>
    constexpr PIKA_FORCEINLINE in_out_out_result<Ts...> make_in_out_out_result(
        pika::tuple<Ts...>&& t)
    {
        static_assert(pika::tuple_size<pika::tuple<Ts...>>::value == 3,
            "size of tuple should be 3 to convert to in_out_out_result");

        using result_type = in_out_out_result<Ts...>;

        return result_type{pika::get<0>(t), pika::get<1>(t), pika::get<2>(t)};
    }

    template <typename... Ts>
    pika::future<in_out_out_result<Ts...>> make_in_out_out_result(
        pika::future<pika::tuple<Ts...>>&& f)
    {
        static_assert(pika::tuple_size<pika::tuple<Ts...>>::value == 3,
            "size of tuple should be 3 to convert to in_out_out_result");

        using result_type = in_out_out_result<Ts...>;

        return pika::make_future<result_type>(
            PIKA_MOVE(f), [](pika::tuple<Ts...>&& t) -> result_type {
                return make_in_out_out_result(PIKA_MOVE(t));
            });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename I, typename F>
    struct in_fun_result
    {
        PIKA_NO_UNIQUE_ADDRESS I in;
        PIKA_NO_UNIQUE_ADDRESS F fun;

        template <typename I2, typename F2,
            typename Enable =
                std::enable_if_t<std::is_convertible_v<I const&, I2> &&
                    std::is_convertible_v<F const&, F2>>>
        constexpr operator in_fun_result<I2, F2>() const&
        {
            return {in, fun};
        }

        template <typename I2, typename F2,
            typename Enable = std::enable_if_t<std::is_convertible_v<I, I2> &&
                std::is_convertible_v<F, F2>>>
        constexpr operator in_fun_result<I2, F2>() &&
        {
            return {PIKA_MOVE(in), PIKA_MOVE(fun)};
        }

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            // clang-format off
            ar & in & fun;
            // clang-format on
        }
    };

    template <typename Iterator, typename Sentinel = Iterator>
    pika::util::iterator_range<Iterator, Sentinel> make_subrange(
        Iterator iterator, Sentinel sentinel)
    {
        return pika::util::make_iterator_range<Iterator, Sentinel>(
            iterator, sentinel);
    }

    template <typename Iterator, typename Sentinel = Iterator>
    pika::future<pika::util::iterator_range<Iterator, Sentinel>> make_subrange(
        pika::future<Iterator>&& iterator, Sentinel sentinel)
    {
        return pika::make_future<pika::util::iterator_range<Iterator, Sentinel>>(
            PIKA_MOVE(iterator), [sentinel](Iterator&& it) {
                return pika::util::iterator_range<Iterator, Sentinel>(
                    it, sentinel);
            });
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename ZipIter>
        in_out_result<typename pika::tuple_element<0,
                          typename ZipIter::iterator_tuple_type>::type,
            typename pika::tuple_element<1,
                typename ZipIter::iterator_tuple_type>::type>
        get_in_out_result(ZipIter&& zipiter)
        {
            using iterator_tuple_type = typename ZipIter::iterator_tuple_type;

            using result_type = in_out_result<
                typename pika::tuple_element<0, iterator_tuple_type>::type,
                typename pika::tuple_element<1, iterator_tuple_type>::type>;

            iterator_tuple_type t = zipiter.get_iterator_tuple();
            return result_type{pika::get<0>(t), pika::get<1>(t)};
        }

        template <typename ZipIter>
        pika::future<
            in_out_result<typename pika::tuple_element<0,
                              typename ZipIter::iterator_tuple_type>::type,
                typename pika::tuple_element<1,
                    typename ZipIter::iterator_tuple_type>::type>>
        get_in_out_result(pika::future<ZipIter>&& zipiter)
        {
            using iterator_tuple_type = typename ZipIter::iterator_tuple_type;

            using result_type = in_out_result<
                typename pika::tuple_element<0, iterator_tuple_type>::type,
                typename pika::tuple_element<1, iterator_tuple_type>::type>;

            return pika::make_future<result_type>(
                PIKA_MOVE(zipiter), [](ZipIter zipiter) {
                    return get_in_out_result(PIKA_MOVE(zipiter));
                });
        }

        template <typename ZipIter>
        min_max_result<typename pika::tuple_element<0,
            typename ZipIter::iterator_tuple_type>::type>
        get_min_max_result(ZipIter&& zipiter)
        {
            using iterator_tuple_type = typename ZipIter::iterator_tuple_type;

            using result_type = min_max_result<
                typename pika::tuple_element<0, iterator_tuple_type>::type>;

            iterator_tuple_type t = zipiter.get_iterator_tuple();
            return result_type{pika::get<0>(t), pika::get<1>(t)};
        }

        template <typename ZipIter>
        pika::future<min_max_result<typename pika::tuple_element<0,
            typename ZipIter::iterator_tuple_type>::type>>
        get_min_max_result(pika::future<ZipIter>&& zipiter)
        {
            using iterator_tuple_type = typename ZipIter::iterator_tuple_type;

            using result_type = min_max_result<
                typename pika::tuple_element<0, iterator_tuple_type>::type>;

            return pika::make_future<result_type>(
                PIKA_MOVE(zipiter), [](ZipIter zipiter) {
                    return get_min_max_result(PIKA_MOVE(zipiter));
                });
        }

        template <typename ZipIter>
        in_in_out_result<typename pika::tuple_element<0,
                             typename ZipIter::iterator_tuple_type>::type,
            typename pika::tuple_element<1,
                typename ZipIter::iterator_tuple_type>::type,
            typename pika::tuple_element<2,
                typename ZipIter::iterator_tuple_type>::type>
        get_in_in_out_result(ZipIter&& zipiter)
        {
            using iterator_tuple_type = typename ZipIter::iterator_tuple_type;

            using result_type = in_in_out_result<
                typename pika::tuple_element<0, iterator_tuple_type>::type,
                typename pika::tuple_element<1, iterator_tuple_type>::type,
                typename pika::tuple_element<2, iterator_tuple_type>::type>;

            iterator_tuple_type t = zipiter.get_iterator_tuple();
            return result_type{pika::get<0>(t), pika::get<1>(t), pika::get<2>(t)};
        }

        template <typename ZipIter>
        pika::future<
            in_in_out_result<typename pika::tuple_element<0,
                                 typename ZipIter::iterator_tuple_type>::type,
                typename pika::tuple_element<1,
                    typename ZipIter::iterator_tuple_type>::type,
                typename pika::tuple_element<2,
                    typename ZipIter::iterator_tuple_type>::type>>
        get_in_in_out_result(pika::future<ZipIter>&& zipiter)
        {
            using iterator_tuple_type = typename ZipIter::iterator_tuple_type;

            using result_type = in_in_out_result<
                typename pika::tuple_element<0, iterator_tuple_type>::type,
                typename pika::tuple_element<1, iterator_tuple_type>::type,
                typename pika::tuple_element<2, iterator_tuple_type>::type>;

            return pika::make_future<result_type>(
                PIKA_MOVE(zipiter), [](ZipIter zipiter) {
                    return get_in_in_out_result(PIKA_MOVE(zipiter));
                });
        }
    }    // namespace detail
}}}      // namespace pika::parallel::util

namespace pika { namespace ranges {
    using pika::parallel::util::in_fun_result;
    using pika::parallel::util::in_in_out_result;
    using pika::parallel::util::in_in_result;
    using pika::parallel::util::in_out_out_result;
    using pika::parallel::util::in_out_result;
    using pika::parallel::util::min_max_result;
}}    // namespace pika::ranges
