//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/execution/traits/is_execution_policy.hpp>
#include <pika/functional/detail/invoke.hpp>
#include <pika/parallel/util/cancellation_token.hpp>
#include <pika/parallel/util/result_types.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pika::parallel::detail {
    template <typename Iter>
    struct transform_loop_impl
    {
        template <typename InIterB, typename InIterE, typename OutIter,
            typename F>
        PIKA_HOST_DEVICE
            PIKA_FORCEINLINE static constexpr in_out_result<InIterB, OutIter>
            call(InIterB first, InIterE last, OutIter dest, F&& f)
        {
            for (/* */; first != last; (void) ++first, ++dest)
            {
                *dest = PIKA_INVOKE(f, first);
            }

            return in_out_result<InIterB, OutIter>{
                PIKA_MOVE(first), PIKA_MOVE(dest)};
        }
    };

    struct transform_loop_t final
      : pika::functional::detail::tag_fallback<transform_loop_t>
    {
    private:
        template <typename ExPolicy, typename IterB, typename IterE,
            typename OutIter, typename F>
        friend PIKA_HOST_DEVICE
            PIKA_FORCEINLINE constexpr in_out_result<IterB, OutIter>
            tag_fallback_invoke(transform_loop_t, ExPolicy&&, IterB it,
                IterE end, OutIter dest, F&& f)
        {
            return transform_loop_impl<IterB>::call(
                it, end, dest, PIKA_FORWARD(F, f));
        }
    };

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
    inline constexpr transform_loop_t transform_loop = transform_loop_t{};
#else
    template <typename ExPolicy, typename IterB, typename IterE,
        typename OutIter, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr in_out_result<IterB, OutIter>
    transform_loop(ExPolicy&& policy, IterB it, IterE end, OutIter dest, F&& f)
    {
        return transform_loop_t{}(
            PIKA_FORWARD(ExPolicy, policy), it, end, dest, PIKA_FORWARD(F, f));
    }
#endif

    template <typename Iter>
    struct transform_loop_ind_impl
    {
        template <typename InIterB, typename InIterE, typename OutIter,
            typename F>
        PIKA_HOST_DEVICE
            PIKA_FORCEINLINE static constexpr in_out_result<InIterB, OutIter>
            call(InIterB first, InIterE last, OutIter dest, F&& f)
        {
            for (/* */; first != last; (void) ++first, ++dest)
            {
                *dest = PIKA_INVOKE(f, *first);
            }

            return in_out_result<InIterB, OutIter>{
                PIKA_MOVE(first), PIKA_MOVE(dest)};
        }
    };

    struct transform_loop_ind_t final
      : pika::functional::detail::tag_fallback<transform_loop_ind_t>
    {
    private:
        template <typename ExPolicy, typename IterB, typename IterE,
            typename OutIter, typename F>
        friend PIKA_HOST_DEVICE
            PIKA_FORCEINLINE constexpr in_out_result<IterB, OutIter>
            tag_fallback_invoke(transform_loop_ind_t, ExPolicy&&, IterB it,
                IterE end, OutIter dest, F&& f)
        {
            return transform_loop_ind_impl<IterB>::call(
                it, end, dest, PIKA_FORWARD(F, f));
        }
    };

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
    inline constexpr transform_loop_ind_t transform_loop_ind =
        transform_loop_ind_t{};
#else
    template <typename ExPolicy, typename IterB, typename IterE,
        typename OutIter, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr in_out_result<IterB, OutIter>
    transform_loop_ind(
        ExPolicy&& policy, IterB it, IterE end, OutIter dest, F&& f)
    {
        return transform_loop_ind_t{}(
            PIKA_FORWARD(ExPolicy, policy), it, end, dest, PIKA_FORWARD(F, f));
    }
#endif

    template <typename Iter1, typename Iter2>
    struct transform_binary_loop_impl
    {
        template <typename InIter1B, typename InIter1E, typename InIter2,
            typename OutIter, typename F>
        PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr in_in_out_result<
            InIter1B, InIter2, OutIter>
        call(InIter1B first1, InIter1E last1, InIter2 first2, OutIter dest,
            F&& f)
        {
            for (/* */; first1 != last1; (void) ++first1, ++first2, ++dest)
            {
                *dest = PIKA_INVOKE(f, first1, first2);
            }

            return in_in_out_result<InIter1B, InIter2, OutIter>{
                PIKA_MOVE(first1), PIKA_MOVE(first2), PIKA_MOVE(dest)};
        }

        template <typename InIter1B, typename InIter1E, typename InIter2,
            typename OutIter, typename F>
        PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr in_in_out_result<
            InIter1B, InIter2, OutIter>
        call(InIter1B first1, InIter1E last1, InIter2 first2, InIter2 last2,
            OutIter dest, F&& f)
        {
            for (/* */; first1 != last1 && first2 != last2;
                 (void) ++first1, ++first2, ++dest)
            {
                *dest = PIKA_INVOKE(f, first1, first2);
            }

            return in_in_out_result<InIter1B, InIter2, OutIter>{
                first1, first2, dest};
        }
    };

    template <typename ExPolicy>
    struct transform_binary_loop_t final
      : pika::functional::detail::tag_fallback<
            transform_binary_loop_t<ExPolicy>>
    {
    private:
        template <typename InIter1B, typename InIter1E, typename InIter2,
            typename OutIter, typename F>
        friend PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr in_in_out_result<
            InIter1B, InIter2, OutIter>
        tag_fallback_invoke(transform_binary_loop_t<ExPolicy>, InIter1B first1,
            InIter1E last1, InIter2 first2, OutIter dest, F&& f)
        {
            return transform_binary_loop_impl<InIter1B, InIter2>::call(
                first1, last1, first2, dest, PIKA_FORWARD(F, f));
        }

        template <typename InIter1B, typename InIter1E, typename InIter2B,
            typename InIter2E, typename OutIter, typename F>
        friend PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr in_in_out_result<
            InIter1B, InIter2B, OutIter>
        tag_fallback_invoke(transform_binary_loop_t<ExPolicy>, InIter1B first1,
            InIter1E last1, InIter2B first2, InIter2E last2, OutIter dest,
            F&& f)
        {
            return transform_binary_loop_impl<InIter1B, InIter2B>::call(
                first1, last1, first2, last2, dest, PIKA_FORWARD(F, f));
        }
    };

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr transform_binary_loop_t<ExPolicy> transform_binary_loop =
        transform_binary_loop_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename InIter1B, typename InIter1E,
        typename InIter2, typename OutIter, typename F>
    PIKA_HOST_DEVICE
        PIKA_FORCEINLINE constexpr in_in_out_result<InIter1B, InIter2, OutIter>
        transform_binary_loop(InIter1B first1, InIter1E last1, InIter2 first2,
            OutIter dest, F&& f)
    {
        return transform_binary_loop_t<ExPolicy>{}(
            first1, last1, first2, dest, PIKA_FORWARD(F, f));
    }

    template <typename ExPolicy, typename InIter1B, typename InIter1E,
        typename InIter2B, typename InIter2E, typename OutIter, typename F>
    PIKA_HOST_DEVICE
        PIKA_FORCEINLINE constexpr in_in_out_result<InIter1B, InIter2B, OutIter>
        transform_binary_loop(InIter1B first1, InIter1E last1, InIter2B first2,
            InIter2E last2, OutIter dest, F&& f)
    {
        return transform_binary_loop_t<ExPolicy>{}(
            first1, last1, first2, last2, dest, PIKA_FORWARD(F, f));
    }
#endif

    template <typename Iter1, typename Iter2>
    struct transform_binary_loop_ind_impl
    {
        template <typename InIter1B, typename InIter1E, typename InIter2,
            typename OutIter, typename F>
        PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr in_in_out_result<
            InIter1B, InIter2, OutIter>
        call(InIter1B first1, InIter1E last1, InIter2 first2, OutIter dest,
            F&& f)
        {
            for (/* */; first1 != last1; (void) ++first1, ++first2, ++dest)
            {
                *dest = PIKA_INVOKE(f, *first1, *first2);
            }

            return in_in_out_result<InIter1B, InIter2, OutIter>{
                PIKA_MOVE(first1), PIKA_MOVE(first2), PIKA_MOVE(dest)};
        }

        template <typename InIter1B, typename InIter1E, typename InIter2,
            typename OutIter, typename F>
        PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr in_in_out_result<
            InIter1B, InIter2, OutIter>
        call(InIter1B first1, InIter1E last1, InIter2 first2, InIter2 last2,
            OutIter dest, F&& f)
        {
            for (/* */; first1 != last1 && first2 != last2;
                 (void) ++first1, ++first2, ++dest)
            {
                *dest = PIKA_INVOKE(f, *first1, *first2);
            }

            return in_in_out_result<InIter1B, InIter2, OutIter>{
                first1, first2, dest};
        }
    };

    template <typename ExPolicy>
    struct transform_binary_loop_ind_t final
      : pika::functional::detail::tag_fallback<
            transform_binary_loop_ind_t<ExPolicy>>
    {
    private:
        template <typename InIter1B, typename InIter1E, typename InIter2,
            typename OutIter, typename F>
        friend PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr in_in_out_result<
            InIter1B, InIter2, OutIter>
        tag_fallback_invoke(transform_binary_loop_ind_t<ExPolicy>,
            InIter1B first1, InIter1E last1, InIter2 first2, OutIter dest,
            F&& f)
        {
            return transform_binary_loop_ind_impl<InIter1B, InIter2>::call(
                first1, last1, first2, dest, PIKA_FORWARD(F, f));
        }

        template <typename InIter1B, typename InIter1E, typename InIter2B,
            typename InIter2E, typename OutIter, typename F>
        friend PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr in_in_out_result<
            InIter1B, InIter2B, OutIter>
        tag_fallback_invoke(transform_binary_loop_ind_t<ExPolicy>,
            InIter1B first1, InIter1E last1, InIter2B first2, InIter2E last2,
            OutIter dest, F&& f)
        {
            return transform_binary_loop_ind_impl<InIter1B, InIter2B>::call(
                first1, last1, first2, last2, dest, PIKA_FORWARD(F, f));
        }
    };

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr transform_binary_loop_ind_t<ExPolicy>
        transform_binary_loop_ind = transform_binary_loop_ind_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename InIter1B, typename InIter1E,
        typename InIter2, typename OutIter, typename F>
    PIKA_HOST_DEVICE
        PIKA_FORCEINLINE constexpr in_in_out_result<InIter1B, InIter2, OutIter>
        transform_binary_loop_ind(InIter1B first1, InIter1E last1,
            InIter2 first2, OutIter dest, F&& f)
    {
        return transform_binary_loop_ind_t<ExPolicy>{}(
            first1, last1, first2, dest, PIKA_FORWARD(F, f));
    }

    template <typename ExPolicy, typename InIter1B, typename InIter1E,
        typename InIter2B, typename InIter2E, typename OutIter, typename F>
    PIKA_HOST_DEVICE
        PIKA_FORCEINLINE constexpr in_in_out_result<InIter1B, InIter2B, OutIter>
        transform_binary_loop_ind(InIter1B first1, InIter1E last1,
            InIter2B first2, InIter2E last2, OutIter dest, F&& f)
    {
        return transform_binary_loop_ind_t<ExPolicy>{}(
            first1, last1, first2, last2, dest, PIKA_FORWARD(F, f));
    }
#endif

    template <typename Iter>
    struct transform_loop_n_impl
    {
        template <typename InIter, typename OutIter, typename F>
        PIKA_HOST_DEVICE
            PIKA_FORCEINLINE static constexpr std::pair<InIter, OutIter>
            call(InIter it, std::size_t num, OutIter dest, F&& f,
                std::false_type)
        {
            std::size_t count(num & std::size_t(-4));                  // -V112
            for (std::size_t i = 0; i < count; (void) ++it, i += 4)    // -V112
            {
                *dest++ = PIKA_INVOKE(f, it);
                *dest++ = PIKA_INVOKE(f, ++it);
                *dest++ = PIKA_INVOKE(f, ++it);
                *dest++ = PIKA_INVOKE(f, ++it);
            }
            for (/**/; count < num; (void) ++count, ++it, ++dest)
            {
                *dest = PIKA_INVOKE(f, it);
            }

            return std::make_pair(PIKA_MOVE(it), PIKA_MOVE(dest));
        }

        template <typename InIter, typename OutIter, typename F>
        PIKA_HOST_DEVICE
            PIKA_FORCEINLINE static constexpr std::pair<InIter, OutIter>
            call(
                InIter it, std::size_t num, OutIter dest, F&& f, std::true_type)
        {
            while (num >= 4)
            {
                *dest++ = PIKA_INVOKE(f, it);
                *dest++ = PIKA_INVOKE(f, it + 1);
                *dest++ = PIKA_INVOKE(f, it + 2);
                *dest++ = PIKA_INVOKE(f, it + 3);

                it += 4;
                num -= 4;
            }

            switch (num)
            {
            case 3:
                *dest++ = PIKA_INVOKE(f, it);
                *dest++ = PIKA_INVOKE(f, it + 1);
                *dest++ = PIKA_INVOKE(f, it + 2);
                break;

            case 2:
                *dest++ = PIKA_INVOKE(f, it);
                *dest++ = PIKA_INVOKE(f, it + 1);
                break;

            case 1:
                *dest++ = PIKA_INVOKE(f, it);
                break;

            default:
                break;
            }

            return std::make_pair(it + num, PIKA_MOVE(dest));
        }
    };

    template <typename ExPolicy>
    struct transform_loop_n_t final
      : pika::functional::detail::tag_fallback<transform_loop_n_t<ExPolicy>>
    {
    private:
        template <typename Iter, typename OutIter, typename F>
        friend PIKA_HOST_DEVICE
            PIKA_FORCEINLINE constexpr std::pair<Iter, OutIter>
            tag_fallback_invoke(transform_loop_n_t<ExPolicy>, Iter it,
                std::size_t count, OutIter dest, F&& f)
        {
            using pred = pika::traits::is_random_access_iterator<Iter>;

            return transform_loop_n_impl<Iter>::call(
                it, count, dest, PIKA_FORWARD(F, f), pred());
        }
    };

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr transform_loop_n_t<ExPolicy> transform_loop_n =
        transform_loop_n_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename Iter, typename OutIter, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr std::pair<Iter, OutIter>
    transform_loop_n(Iter it, std::size_t count, OutIter dest, F&& f)
    {
        return transform_loop_n_t<ExPolicy>{}(
            it, count, dest, PIKA_FORWARD(F, f));
    }
#endif

    template <typename Iter>
    struct transform_loop_n_ind_impl
    {
        template <typename InIter, typename OutIter, typename F>
        PIKA_HOST_DEVICE
            PIKA_FORCEINLINE static constexpr std::pair<InIter, OutIter>
            call(InIter it, std::size_t num, OutIter dest, F&& f,
                std::false_type)
        {
            std::size_t count(num & std::size_t(-4));                  // -V112
            for (std::size_t i = 0; i < count; (void) ++it, i += 4)    // -V112
            {
                *dest++ = PIKA_INVOKE(f, *it);
                *dest++ = PIKA_INVOKE(f, *(++it));
                *dest++ = PIKA_INVOKE(f, *(++it));
                *dest++ = PIKA_INVOKE(f, *(++it));
            }
            for (/**/; count < num; (void) ++count, ++it, ++dest)
            {
                *dest = PIKA_INVOKE(f, *it);
            }

            return std::make_pair(PIKA_MOVE(it), PIKA_MOVE(dest));
        }

        template <typename InIter, typename OutIter, typename F>
        PIKA_HOST_DEVICE
            PIKA_FORCEINLINE static constexpr std::pair<InIter, OutIter>
            call(
                InIter it, std::size_t num, OutIter dest, F&& f, std::true_type)
        {
            while (num >= 4)
            {
                *dest++ = PIKA_INVOKE(f, *it);
                *dest++ = PIKA_INVOKE(f, *(it + 1));
                *dest++ = PIKA_INVOKE(f, *(it + 2));
                *dest++ = PIKA_INVOKE(f, *(it + 3));

                it += 4;
                num -= 4;
            }

            switch (num)
            {
            case 3:
                *dest++ = PIKA_INVOKE(f, *it);
                *dest++ = PIKA_INVOKE(f, *(it + 1));
                *dest++ = PIKA_INVOKE(f, *(it + 2));
                break;

            case 2:
                *dest++ = PIKA_INVOKE(f, *it);
                *dest++ = PIKA_INVOKE(f, *(it + 1));
                break;

            case 1:
                *dest++ = PIKA_INVOKE(f, *it);
                break;

            default:
                break;
            }

            return std::make_pair(it + num, PIKA_MOVE(dest));
        }
    };

    template <typename ExPolicy>
    struct transform_loop_n_ind_t final
      : pika::functional::detail::tag_fallback<transform_loop_n_ind_t<ExPolicy>>
    {
    private:
        template <typename Iter, typename OutIter, typename F>
        friend PIKA_HOST_DEVICE
            PIKA_FORCEINLINE constexpr std::pair<Iter, OutIter>
            tag_fallback_invoke(transform_loop_n_ind_t<ExPolicy>, Iter it,
                std::size_t count, OutIter dest, F&& f)
        {
            using pred = pika::traits::is_random_access_iterator<Iter>;

            return transform_loop_n_ind_impl<Iter>::call(
                it, count, dest, PIKA_FORWARD(F, f), pred());
        }
    };

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr transform_loop_n_ind_t<ExPolicy> transform_loop_n_ind =
        transform_loop_n_ind_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename Iter, typename OutIter, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr std::pair<Iter, OutIter>
    transform_loop_n_ind(Iter it, std::size_t count, OutIter dest, F&& f)
    {
        return transform_loop_n_ind_t<ExPolicy>{}(
            it, count, dest, PIKA_FORWARD(F, f));
    }
#endif

    template <typename Iter1, typename Inter2>
    struct transform_binary_loop_n_impl
    {
        template <typename InIter1, typename InIter2, typename OutIter,
            typename F>
        PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr std::tuple<InIter1,
            InIter2, OutIter>
        call(InIter1 first1, std::size_t num, InIter2 first2, OutIter dest,
            F&& f)
        {
            std::size_t count(num & std::size_t(-4));    // -V112
            for (std::size_t i = 0; i < count;
                 (void) ++first1, ++first2, i += 4)    // -V112
            {
                *dest++ = PIKA_INVOKE(f, first1, first2);
                *dest++ = PIKA_INVOKE(f, ++first1, ++first2);
                *dest++ = PIKA_INVOKE(f, ++first1, ++first2);
                *dest++ = PIKA_INVOKE(f, ++first1, ++first2);
            }
            for (/**/; count < num; (void) ++count, ++first1, ++first2, ++dest)
            {
                *dest = PIKA_INVOKE(f, first1, first2);
            }

            return std::make_tuple(
                PIKA_MOVE(first1), PIKA_MOVE(first2), PIKA_MOVE(dest));
        }
    };

    template <typename ExPolicy>
    struct transform_binary_loop_n_t final
      : pika::functional::detail::tag_fallback<
            transform_binary_loop_n_t<ExPolicy>>
    {
    private:
        template <typename InIter1, typename InIter2, typename OutIter,
            typename F>
        friend PIKA_HOST_DEVICE
            PIKA_FORCEINLINE constexpr std::tuple<InIter1, InIter2, OutIter>
            tag_fallback_invoke(transform_binary_loop_n_t<ExPolicy>,
                InIter1 first1, std::size_t count, InIter2 first2, OutIter dest,
                F&& f)
        {
            return transform_binary_loop_n_impl<InIter1, InIter2>::call(
                first1, count, first2, dest, PIKA_FORWARD(F, f));
        }
    };

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr transform_binary_loop_n_t<ExPolicy>
        transform_binary_loop_n = transform_binary_loop_n_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    PIKA_HOST_DEVICE
        PIKA_FORCEINLINE constexpr std::tuple<InIter1, InIter2, OutIter>
        transform_binary_loop_n(InIter1 first1, std::size_t count,
            InIter2 first2, OutIter dest, F&& f)
    {
        return transform_binary_loop_n_t<ExPolicy>{}(
            first1, count, first2, dest, PIKA_FORWARD(F, f));
    }
#endif

    template <typename Iter1, typename Inter2>
    struct transform_binary_loop_ind_n_impl
    {
        template <typename InIter1, typename InIter2, typename OutIter,
            typename F>
        PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr std::tuple<InIter1,
            InIter2, OutIter>
        call(InIter1 first1, std::size_t num, InIter2 first2, OutIter dest,
            F&& f)
        {
            std::size_t count(num & std::size_t(-4));    // -V112
            for (std::size_t i = 0; i < count;
                 (void) ++first1, ++first2, i += 4)    // -V112
            {
                *dest++ = PIKA_INVOKE(f, *first1, *first2);
                *dest++ = PIKA_INVOKE(f, *(++first1), *(++first2));
                *dest++ = PIKA_INVOKE(f, *(++first1), *(++first2));
                *dest++ = PIKA_INVOKE(f, *(++first1), *(++first2));
            }
            for (/**/; count < num; (void) ++count, ++first1, ++first2, ++dest)
            {
                *dest = PIKA_INVOKE(f, *first1, *first2);
            }

            return std::make_tuple(
                PIKA_MOVE(first1), PIKA_MOVE(first2), PIKA_MOVE(dest));
        }
    };

    template <typename ExPolicy>
    struct transform_binary_loop_ind_n_t final
      : pika::functional::detail::tag_fallback<
            transform_binary_loop_ind_n_t<ExPolicy>>
    {
    private:
        template <typename InIter1, typename InIter2, typename OutIter,
            typename F>
        friend PIKA_HOST_DEVICE
            PIKA_FORCEINLINE constexpr std::tuple<InIter1, InIter2, OutIter>
            tag_fallback_invoke(transform_binary_loop_ind_n_t<ExPolicy>,
                InIter1 first1, std::size_t count, InIter2 first2, OutIter dest,
                F&& f)
        {
            return transform_binary_loop_ind_n_impl<InIter1, InIter2>::call(
                first1, count, first2, dest, PIKA_FORWARD(F, f));
        }
    };

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr transform_binary_loop_ind_n_t<ExPolicy>
        transform_binary_loop_ind_n = transform_binary_loop_ind_n_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    PIKA_HOST_DEVICE
        PIKA_FORCEINLINE constexpr std::tuple<InIter1, InIter2, OutIter>
        transform_binary_loop_ind_n(InIter1 first1, std::size_t count,
            InIter2 first2, OutIter dest, F&& f)
    {
        return transform_binary_loop_ind_n_t<ExPolicy>{}(
            first1, count, first2, dest, PIKA_FORWARD(F, f));
    }
#endif
}    // namespace pika::parallel::detail
