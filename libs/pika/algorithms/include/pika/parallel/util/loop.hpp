//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/execution/traits/is_execution_policy.hpp>
#include <pika/functional/detail/invoke.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/invoke_result.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/parallel/util/cancellation_token.hpp>
#include <pika/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika { namespace parallel { namespace util {

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct loop_step_t final
      : pika::functional::detail::tag_fallback<loop_step_t<ExPolicy>>
    {
    private:
        template <typename VecOnly, typename F, typename... Iters>
        friend PIKA_HOST_DEVICE PIKA_FORCEINLINE
            typename pika::util::invoke_result<F, Iters...>::type
            tag_fallback_invoke(pika::parallel::util::loop_step_t<ExPolicy>,
                VecOnly&&, F&& f, Iters&... its)
        {
            return PIKA_INVOKE(PIKA_FORWARD(F, f), (its++)...);
        }
    };

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr loop_step_t<ExPolicy> loop_step = loop_step_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename VecOnly, typename F,
        typename... Iters>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE
        typename pika::util::invoke_result<F, Iters...>::type
        loop_step(VecOnly&& v, F&& f, Iters&... its)
    {
        return pika::parallel::util::loop_step_t<ExPolicy>{}(
            PIKA_FORWARD(VecOnly, v), PIKA_FORWARD(F, f), (its)...);
    }
#endif

    template <typename ExPolicy>
    struct loop_optimization_t final
      : pika::functional::detail::tag_fallback<loop_optimization_t<ExPolicy>>
    {
    private:
        template <typename Iter>
        friend PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr bool
            tag_fallback_invoke(
                pika::parallel::util::loop_optimization_t<ExPolicy>, Iter, Iter)
        {
            return false;
        }
    };

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr loop_optimization_t<ExPolicy> loop_optimization =
        loop_optimization_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename Iter>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr bool loop_optimization(
        Iter it1, Iter it2)
    {
        return pika::parallel::util::loop_optimization_t<ExPolicy>{}(it1, it2);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // Helper class to repeatedly call a function starting from a given
        // iterator position.
        template <typename Iterator>
        struct loop
        {
            ///////////////////////////////////////////////////////////////////
            template <typename Begin, typename End, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr Begin call(
                Begin it, End end, F&& f)
            {
                for (/**/; it != end; ++it)
                {
                    PIKA_INVOKE(f, it);
                }
                return it;
            }

            template <typename Begin, typename End, typename CancelToken,
                typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr Begin call(
                Begin it, End end, CancelToken& tok, F&& f)
            {
                for (/**/; it != end; ++it)
                {
                    if (tok.was_cancelled())
                        break;
                    PIKA_INVOKE(f, it);
                }
                return it;
            }
        };
    }    // namespace detail

    struct loop_t final : pika::functional::detail::tag_fallback<loop_t>
    {
    private:
        template <typename ExPolicy, typename Begin, typename End, typename F>
        friend PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr Begin
        tag_fallback_invoke(pika::parallel::util::loop_t, ExPolicy&&,
            Begin begin, End end, F&& f)
        {
            return detail::loop<Begin>::call(begin, end, PIKA_FORWARD(F, f));
        }

        template <typename ExPolicy, typename Begin, typename End,
            typename CancelToken, typename F>
        friend PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr Begin
        tag_fallback_invoke(pika::parallel::util::loop_t, ExPolicy&&,
            Begin begin, End end, CancelToken& tok, F&& f)
        {
            return detail::loop<Begin>::call(
                begin, end, tok, PIKA_FORWARD(F, f));
        }
    };

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
    inline constexpr loop_t loop = loop_t{};
#else
    template <typename ExPolicy, typename Begin, typename End, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr Begin loop(
        ExPolicy&& policy, Begin begin, End end, F&& f)
    {
        return pika::parallel::util::loop_t{}(
            PIKA_FORWARD(ExPolicy, policy), begin, end, PIKA_FORWARD(F, f));
    }

    template <typename ExPolicy, typename Begin, typename End,
        typename CancelToken, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr Begin loop(
        ExPolicy&& policy, Begin begin, End end, CancelToken& tok, F&& f)
    {
        return pika::parallel::util::loop_t{}(
            PIKA_FORWARD(ExPolicy, policy), begin, end, tok, PIKA_FORWARD(F, f));
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // Helper class to repeatedly call a function starting from a given
        // iterator position.
        template <typename Iterator>
        struct loop_ind
        {
            ///////////////////////////////////////////////////////////////////
            template <typename Begin, typename End, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr Begin call(
                Begin it, End end, F&& f)
            {
                for (/**/; it != end; ++it)
                {
                    PIKA_INVOKE(f, *it);
                }
                return it;
            }

            template <typename Begin, typename End, typename CancelToken,
                typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr Begin call(
                Begin it, End end, CancelToken& tok, F&& f)
            {
                for (/**/; it != end; ++it)
                {
                    if (tok.was_cancelled())
                        break;
                    PIKA_INVOKE(f, *it);
                }
                return it;
            }
        };
    }    // namespace detail

    struct loop_ind_t final : pika::functional::detail::tag_fallback<loop_ind_t>
    {
    private:
        template <typename ExPolicy, typename Begin, typename End, typename F>
        friend PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr Begin
        tag_fallback_invoke(pika::parallel::util::loop_ind_t, ExPolicy&&,
            Begin begin, End end, F&& f)
        {
            return detail::loop_ind<Begin>::call(begin, end, PIKA_FORWARD(F, f));
        }

        template <typename ExPolicy, typename Begin, typename End,
            typename CancelToken, typename F>
        friend PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr Begin
        tag_fallback_invoke(pika::parallel::util::loop_ind_t, ExPolicy&&,
            Begin begin, End end, CancelToken& tok, F&& f)
        {
            return detail::loop_ind<Begin>::call(
                begin, end, tok, PIKA_FORWARD(F, f));
        }
    };

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
    inline constexpr loop_ind_t loop_ind = loop_ind_t{};
#else
    template <typename ExPolicy, typename Begin, typename End, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr Begin loop_ind(
        ExPolicy&& policy, Begin begin, End end, F&& f)
    {
        return pika::parallel::util::loop_ind_t{}(
            PIKA_FORWARD(ExPolicy, policy), begin, end, PIKA_FORWARD(F, f));
    }

    template <typename ExPolicy, typename Begin, typename End,
        typename CancelToken, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr Begin loop_ind(
        ExPolicy&& policy, Begin begin, End end, CancelToken& tok, F&& f)
    {
        return pika::parallel::util::loop_ind_t{}(
            PIKA_FORWARD(ExPolicy, policy), begin, end, tok, PIKA_FORWARD(F, f));
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // Helper class to repeatedly call a function starting from a given
        // iterator position.
        template <typename Iter1, typename Iter2>
        struct loop2
        {
            ///////////////////////////////////////////////////////////////////
            template <typename Begin1, typename End1, typename Begin2,
                typename F>
            PIKA_HOST_DEVICE
                PIKA_FORCEINLINE static constexpr std::pair<Begin1, Begin2>
                call(Begin1 it1, End1 end1, Begin2 it2, F&& f)
            {
                for (/**/; it1 != end1; (void) ++it1, ++it2)
                {
                    PIKA_INVOKE(f, it1, it2);
                }

                return std::make_pair(PIKA_MOVE(it1), PIKA_MOVE(it2));
            }
        };
    }    // namespace detail

    template <typename ExPolicy>
    struct loop2_t final
      : pika::functional::detail::tag_fallback<loop2_t<ExPolicy>>
    {
    private:
        template <typename VecOnly, typename Begin1, typename End1,
            typename Begin2, typename F>
        friend PIKA_HOST_DEVICE
            PIKA_FORCEINLINE constexpr std::pair<Begin1, Begin2>
            tag_fallback_invoke(pika::parallel::util::loop2_t<ExPolicy>,
                VecOnly&&, Begin1 begin1, End1 end1, Begin2 begin2, F&& f)
        {
            return detail::loop2<Begin1, Begin2>::call(
                begin1, end1, begin2, PIKA_FORWARD(F, f));
        }
    };

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr loop2_t<ExPolicy> loop2 = loop2_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename VecOnly, typename Begin1,
        typename End1, typename Begin2, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr std::pair<Begin1, Begin2> loop2(
        VecOnly&& v, Begin1 begin1, End1 end1, Begin2 begin2, F&& f)
    {
        return pika::parallel::util::loop2_t<ExPolicy>{}(
            PIKA_FORWARD(VecOnly, v), begin1, end1, begin2, PIKA_FORWARD(F, f));
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // Helper class to repeatedly call a function a given number of times
        // starting from a given iterator position.
        struct loop_n_helper
        {
            ///////////////////////////////////////////////////////////////////
            // handle sequences of non-futures
            template <typename Iter, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr Iter call(
                Iter it, std::size_t num, F&& f, std::false_type)
            {
                std::size_t count(num & std::size_t(-4));    // -V112
                for (std::size_t i = 0; i < count;
                     (void) ++it, i += 4)    // -V112
                {
                    PIKA_INVOKE(f, it);
                    PIKA_INVOKE(f, ++it);
                    PIKA_INVOKE(f, ++it);
                    PIKA_INVOKE(f, ++it);
                }
                for (/**/; count < num; (void) ++count, ++it)
                {
                    PIKA_INVOKE(f, it);
                }
                return it;
            }

            template <typename Iter, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr Iter call(
                Iter it, std::size_t num, F&& f, std::true_type)
            {
                while (num >= 4)
                {
                    PIKA_INVOKE(f, it);
                    PIKA_INVOKE(f, it + 1);
                    PIKA_INVOKE(f, it + 2);
                    PIKA_INVOKE(f, it + 3);

                    it += 4;
                    num -= 4;
                }

                switch (num)
                {
                case 3:
                    PIKA_INVOKE(f, it);
                    PIKA_INVOKE(f, it + 1);
                    PIKA_INVOKE(f, it + 2);
                    break;

                case 2:
                    PIKA_INVOKE(f, it);
                    PIKA_INVOKE(f, it + 1);
                    break;

                case 1:
                    PIKA_INVOKE(f, it);
                    break;

                default:
                    break;
                }

                return it + num;
            }

            template <typename Iter, typename CancelToken, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr Iter call(Iter it,
                std::size_t num, CancelToken& tok, F&& f, std::false_type)
            {
                std::size_t count(num & std::size_t(-4));    // -V112
                for (std::size_t i = 0; i < count;
                     (void) ++it, i += 4)    // -V112
                {
                    if (tok.was_cancelled())
                        break;
                    PIKA_INVOKE(f, it);
                    PIKA_INVOKE(f, ++it);
                    PIKA_INVOKE(f, ++it);
                    PIKA_INVOKE(f, ++it);
                }
                for (/**/; count < num; (void) ++count, ++it)
                {
                    if (tok.was_cancelled())
                        break;
                    PIKA_INVOKE(f, it);
                }
                return it;
            }

            template <typename Iter, typename CancelToken, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr Iter call(Iter it,
                std::size_t num, CancelToken& tok, F&& f, std::true_type)
            {
                while (num >= 4)
                {
                    if (tok.was_cancelled())
                        return it;

                    PIKA_INVOKE(f, it);
                    PIKA_INVOKE(f, it + 1);
                    PIKA_INVOKE(f, it + 2);
                    PIKA_INVOKE(f, it + 3);

                    it += 4;
                    num -= 4;
                }

                switch (num)
                {
                case 3:
                    PIKA_INVOKE(f, it);
                    PIKA_INVOKE(f, it + 1);
                    PIKA_INVOKE(f, it + 2);
                    break;

                case 2:
                    PIKA_INVOKE(f, it);
                    PIKA_INVOKE(f, it + 1);
                    break;

                case 1:
                    PIKA_INVOKE(f, it);
                    break;

                default:
                    break;
                }

                return it + num;
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct loop_n_t final
      : pika::functional::detail::tag_fallback<loop_n_t<ExPolicy>>
    {
    private:
        template <typename Iter, typename F>
        friend PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr Iter
        tag_fallback_invoke(pika::parallel::util::loop_n_t<ExPolicy>, Iter it,
            std::size_t count, F&& f)
        {
            using pred = std::integral_constant<bool,
                pika::traits::is_random_access_iterator<Iter>::value ||
                    std::is_integral<Iter>::value>;

            return detail::loop_n_helper::call(
                it, count, PIKA_FORWARD(F, f), pred());
        }

        template <typename Iter, typename CancelToken, typename F>
        friend PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr Iter
        tag_fallback_invoke(pika::parallel::util::loop_n_t<ExPolicy>, Iter it,
            std::size_t count, CancelToken& tok, F&& f)
        {
            using pred = std::integral_constant<bool,
                pika::traits::is_random_access_iterator<Iter>::value ||
                    std::is_integral<Iter>::value>;

            return detail::loop_n_helper::call(
                it, count, tok, PIKA_FORWARD(F, f), pred());
        }
    };

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr loop_n_t<ExPolicy> loop_n = loop_n_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename Iter, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr Iter loop_n(
        Iter it, std::size_t count, F&& f)
    {
        return pika::parallel::util::loop_n_t<ExPolicy>{}(
            it, count, PIKA_FORWARD(F, f));
    }

    template <typename ExPolicy, typename Iter, typename CancelToken,
        typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr Iter loop_n(
        Iter it, std::size_t count, CancelToken& tok, F&& f)
    {
        return pika::parallel::util::loop_n_t<ExPolicy>{}(
            it, count, tok, PIKA_FORWARD(F, f));
    }
#endif

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy>
        struct extract_value_t
          : pika::functional::detail::tag_fallback<extract_value_t<ExPolicy>>
        {
        private:
            template <typename T>
            friend PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr T const&
            tag_fallback_invoke(
                pika::parallel::util::detail::extract_value_t<ExPolicy>,
                T const& v)
            {
                return v;
            }
        };

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
        template <typename ExPolicy>
        inline constexpr extract_value_t<ExPolicy> extract_value =
            extract_value_t<ExPolicy>{};
#else
        template <typename ExPolicy, typename T>
        PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr T const& extract_value(
            T const& v)
        {
            return pika::parallel::util::detail::extract_value_t<ExPolicy>{}(v);
        }
#endif

        template <typename ExPolicy>
        struct accumulate_values_t
          : pika::functional::detail::tag_fallback<accumulate_values_t<ExPolicy>>
        {
        private:
            template <typename F, typename T>
            friend PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr T const&
            tag_fallback_invoke(
                pika::parallel::util::detail::accumulate_values_t<ExPolicy>, F&&,
                T const& v)
            {
                return v;
            }

            template <typename F, typename T, typename T1>
            friend PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr T
            tag_fallback_invoke(
                pika::parallel::util::detail::accumulate_values_t<ExPolicy>,
                F&& f, T&& v, T1&& init)
            {
                return PIKA_INVOKE(PIKA_FORWARD(F, f), PIKA_FORWARD(T1, init),
                    PIKA_FORWARD(T, v));
            }
        };

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
        template <typename ExPolicy>
        inline constexpr accumulate_values_t<ExPolicy> accumulate_values =
            accumulate_values_t<ExPolicy>{};
#else
        template <typename ExPolicy, typename F, typename T>
        PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr T const& accumulate_values(
            F&& f, T const& v)
        {
            return pika::parallel::util::detail::accumulate_values_t<ExPolicy>{}(
                PIKA_FORWARD(F, f), v);
        }

        template <typename ExPolicy, typename F, typename T, typename T1>
        PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr T accumulate_values(
            F&& f, T&& v, T1&& init)
        {
            return pika::parallel::util::detail::accumulate_values_t<ExPolicy>{}(
                PIKA_FORWARD(F, f), PIKA_FORWARD(T1, v), PIKA_FORWARD(T, init));
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        // Helper class to repeatedly call a function a given number of times
        // starting from a given iterator position.
        struct loop_n_ind_helper
        {
            ///////////////////////////////////////////////////////////////////
            // handle sequences of non-futures
            template <typename Iter, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr Iter call(
                Iter it, std::size_t num, F&& f, std::false_type)
            {
                std::size_t count(num & std::size_t(-4));    // -V112
                for (std::size_t i = 0; i < count;
                     (void) ++it, i += 4)    // -V112
                {
                    PIKA_INVOKE(f, *it);
                    PIKA_INVOKE(f, *(++it));
                    PIKA_INVOKE(f, *(++it));
                    PIKA_INVOKE(f, *(++it));
                }
                for (/**/; count < num; (void) ++count, ++it)
                {
                    PIKA_INVOKE(f, *it);
                }

                return it;
            }

            template <typename Iter, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr Iter call(
                Iter it, std::size_t num, F&& f, std::true_type)
            {
                while (num >= 4)
                {
                    PIKA_INVOKE(f, *it);
                    PIKA_INVOKE(f, *(it + 1));
                    PIKA_INVOKE(f, *(it + 2));
                    PIKA_INVOKE(f, *(it + 3));

                    it += 4;
                    num -= 4;
                }

                switch (num)
                {
                case 3:
                    PIKA_INVOKE(f, *it);
                    PIKA_INVOKE(f, *(it + 1));
                    PIKA_INVOKE(f, *(it + 2));
                    break;

                case 2:
                    PIKA_INVOKE(f, *it);
                    PIKA_INVOKE(f, *(it + 1));
                    break;

                case 1:
                    PIKA_INVOKE(f, *it);
                    break;

                default:
                    break;
                }

                return it + num;
            }

            template <typename Iter, typename CancelToken, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr Iter call(Iter it,
                std::size_t num, CancelToken& tok, F&& f, std::false_type)
            {
                std::size_t count(num & std::size_t(-4));    // -V112
                for (std::size_t i = 0; i < count;
                     (void) ++it, i += 4)    // -V112
                {
                    if (tok.was_cancelled())
                        break;
                    PIKA_INVOKE(f, *it);
                    PIKA_INVOKE(f, *(++it));
                    PIKA_INVOKE(f, *(++it));
                    PIKA_INVOKE(f, *(++it));
                }
                for (/**/; count < num; (void) ++count, ++it)
                {
                    if (tok.was_cancelled())
                        break;
                    PIKA_INVOKE(f, *it);
                }
                return it;
            }

            template <typename Iter, typename CancelToken, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr Iter call(Iter it,
                std::size_t num, CancelToken& tok, F&& f, std::true_type)
            {
                while (num >= 4)
                {
                    if (tok.was_cancelled())
                        return it;

                    PIKA_INVOKE(f, *it);
                    PIKA_INVOKE(f, *(it + 1));
                    PIKA_INVOKE(f, *(it + 2));
                    PIKA_INVOKE(f, *(it + 3));

                    it += 4;
                    num -= 4;
                }

                switch (num)
                {
                case 3:
                    PIKA_INVOKE(f, *it);
                    PIKA_INVOKE(f, *(it + 1));
                    PIKA_INVOKE(f, *(it + 2));
                    break;

                case 2:
                    PIKA_INVOKE(f, *it);
                    PIKA_INVOKE(f, *(it + 1));
                    break;

                case 1:
                    PIKA_INVOKE(f, *it);
                    break;

                default:
                    break;
                }

                return it + num;
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct loop_n_ind_t final
      : pika::functional::detail::tag_fallback<loop_n_ind_t<ExPolicy>>
    {
    private:
        template <typename Iter, typename F>
        friend PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr Iter
        tag_fallback_invoke(pika::parallel::util::loop_n_ind_t<ExPolicy>,
            Iter it, std::size_t count, F&& f)
        {
            using pred = std::integral_constant<bool,
                pika::traits::is_random_access_iterator<Iter>::value ||
                    std::is_integral<Iter>::value>;

            return detail::loop_n_ind_helper::call(
                it, count, PIKA_FORWARD(F, f), pred());
        }

        template <typename Iter, typename CancelToken, typename F>
        friend PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr Iter
        tag_fallback_invoke(pika::parallel::util::loop_n_ind_t<ExPolicy>,
            Iter it, std::size_t count, CancelToken& tok, F&& f)
        {
            using pred = std::integral_constant<bool,
                pika::traits::is_random_access_iterator<Iter>::value ||
                    std::is_integral<Iter>::value>;

            return detail::loop_n_ind_helper::call(
                it, count, tok, PIKA_FORWARD(F, f), pred());
        }
    };

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr loop_n_ind_t<ExPolicy> loop_n_ind =
        loop_n_ind_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename Iter, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr Iter loop_n_ind(
        Iter it, std::size_t count, F&& f)
    {
        return pika::parallel::util::loop_n_ind_t<ExPolicy>{}(
            it, count, PIKA_FORWARD(F, f));
    }

    template <typename ExPolicy, typename Iter, typename CancelToken,
        typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr Iter loop_n_ind(
        Iter it, std::size_t count, CancelToken& tok, F&& f)
    {
        return pika::parallel::util::loop_n_ind_t<ExPolicy>{}(
            it, count, tok, PIKA_FORWARD(F, f));
    }
#endif

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        // Helper class to repeatedly call a function a given number of times
        // starting from a given iterator position. If an exception is thrown,
        // the given cleanup function will be called.
        template <typename IterCat>
        struct loop_with_cleanup
        {
            ///////////////////////////////////////////////////////////////////
            template <typename FwdIter, typename F, typename Cleanup>
            static FwdIter call(
                FwdIter it, FwdIter last, F&& f, Cleanup&& cleanup)
            {
                FwdIter base = it;
                try
                {
                    for (/**/; it != last; ++it)
                    {
                        PIKA_INVOKE(f, it);
                    }
                    return it;
                }
                catch (...)
                {
                    for (/**/; base != it; ++base)
                        cleanup(base);
                    throw;
                }
            }

            template <typename Iter, typename FwdIter, typename F,
                typename Cleanup>
            static FwdIter call(
                Iter it, Iter last, FwdIter dest, F&& f, Cleanup&& cleanup)
            {
                FwdIter base = dest;
                try
                {
                    for (/**/; it != last; (void) ++it, ++dest)
                        f(it, dest);
                    return dest;
                }
                catch (...)
                {
                    for (/**/; base != dest; ++base)
                    {
                        PIKA_INVOKE(cleanup, base);
                    }
                    throw;
                }
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename F, typename Cleanup>
    PIKA_FORCEINLINE constexpr Iter loop_with_cleanup(
        Iter it, Iter last, F&& f, Cleanup&& cleanup)
    {
        using cat = typename std::iterator_traits<Iter>::iterator_category;
        return detail::loop_with_cleanup<cat>::call(
            it, last, PIKA_FORWARD(F, f), PIKA_FORWARD(Cleanup, cleanup));
    }

    template <typename Iter, typename FwdIter, typename F, typename Cleanup>
    PIKA_FORCEINLINE constexpr FwdIter loop_with_cleanup(
        Iter it, Iter last, FwdIter dest, F&& f, Cleanup&& cleanup)
    {
        using cat = typename std::iterator_traits<Iter>::iterator_category;
        return detail::loop_with_cleanup<cat>::call(
            it, last, dest, PIKA_FORWARD(F, f), PIKA_FORWARD(Cleanup, cleanup));
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // Helper class to repeatedly call a function a given number of times
        // starting from a given iterator position.
        template <typename IterCat>
        struct loop_with_cleanup_n
        {
            ///////////////////////////////////////////////////////////////////
            template <typename FwdIter, typename F, typename Cleanup>
            static FwdIter call(
                FwdIter it, std::size_t num, F&& f, Cleanup&& cleanup)
            {
                FwdIter base = it;
                try
                {
                    std::size_t count(num & std::size_t(-4));    // -V112
                    for (std::size_t i = 0; i < count;
                         (void) ++it, i += 4)    // -V112
                    {
                        PIKA_INVOKE(f, it);
                        PIKA_INVOKE(f, ++it);
                        PIKA_INVOKE(f, ++it);
                        PIKA_INVOKE(f, ++it);
                    }
                    for (/**/; count < num; (void) ++count, ++it)
                    {
                        PIKA_INVOKE(f, it);
                    }
                    return it;
                }
                catch (...)
                {
                    for (/**/; base != it; ++base)
                    {
                        PIKA_INVOKE(cleanup, base);
                    }
                    throw;
                }
            }

            template <typename Iter, typename FwdIter, typename F,
                typename Cleanup>
            static FwdIter call(Iter it, std::size_t num, FwdIter dest, F&& f,
                Cleanup&& cleanup)
            {
                FwdIter base = dest;
                try
                {
                    std::size_t count(num & std::size_t(-4));    // -V112
                    for (std::size_t i = 0; i < count;
                         (void) ++it, ++dest, i += 4)    // -V112
                    {
                        PIKA_INVOKE(f, it, dest);
                        PIKA_INVOKE(f, ++it, ++dest);
                        PIKA_INVOKE(f, ++it, ++dest);
                        PIKA_INVOKE(f, ++it, ++dest);
                    }
                    for (/**/; count < num; (void) ++count, ++it, ++dest)
                    {
                        PIKA_INVOKE(f, it, dest);
                    }
                    return dest;
                }
                catch (...)
                {
                    for (/**/; base != dest; ++base)
                    {
                        PIKA_INVOKE(cleanup, base);
                    }
                    throw;
                }
            }

            ///////////////////////////////////////////////////////////////////
            template <typename FwdIter, typename CancelToken, typename F,
                typename Cleanup>
            static FwdIter call_with_token(FwdIter it, std::size_t num,
                CancelToken& tok, F&& f, Cleanup&& cleanup)
            {
                FwdIter base = it;
                try
                {
                    std::size_t count(num & std::size_t(-4));    // -V112
                    for (std::size_t i = 0; i < count;
                         (void) ++it, i += 4)    // -V112
                    {
                        if (tok.was_cancelled())
                            break;

                        PIKA_INVOKE(f, it);
                        PIKA_INVOKE(f, ++it);
                        PIKA_INVOKE(f, ++it);
                        PIKA_INVOKE(f, ++it);
                    }
                    for (/**/; count < num; (void) ++count, ++it)
                    {
                        if (tok.was_cancelled())
                            break;

                        PIKA_INVOKE(f, it);
                    }
                    return it;
                }
                catch (...)
                {
                    tok.cancel();
                    for (/**/; base != it; ++base)
                    {
                        PIKA_INVOKE(cleanup, base);
                    }
                    throw;
                }
            }

            template <typename Iter, typename FwdIter, typename CancelToken,
                typename F, typename Cleanup>
            static FwdIter call_with_token(Iter it, std::size_t num,
                FwdIter dest, CancelToken& tok, F&& f, Cleanup&& cleanup)
            {
                FwdIter base = dest;
                try
                {
                    std::size_t count(num & std::size_t(-4));    // -V112
                    for (std::size_t i = 0; i < count;
                         (void) ++it, ++dest, i += 4)    // -V112
                    {
                        if (tok.was_cancelled())
                            break;

                        PIKA_INVOKE(f, it, dest);
                        PIKA_INVOKE(f, ++it, ++dest);
                        PIKA_INVOKE(f, ++it, ++dest);
                        PIKA_INVOKE(f, ++it, ++dest);
                    }
                    for (/**/; count < num; (void) ++count, ++it, ++dest)
                    {
                        if (tok.was_cancelled())
                            break;

                        PIKA_INVOKE(f, it, dest);
                    }
                    return dest;
                }
                catch (...)
                {
                    tok.cancel();
                    for (/**/; base != dest; ++base)
                    {
                        PIKA_INVOKE(cleanup, base);
                    }
                    throw;
                }
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename F, typename Cleanup>
    PIKA_FORCEINLINE constexpr Iter loop_with_cleanup_n(
        Iter it, std::size_t count, F&& f, Cleanup&& cleanup)
    {
        using cat = typename std::iterator_traits<Iter>::iterator_category;
        return detail::loop_with_cleanup_n<cat>::call(
            it, count, PIKA_FORWARD(F, f), PIKA_FORWARD(Cleanup, cleanup));
    }

    template <typename Iter, typename FwdIter, typename F, typename Cleanup>
    PIKA_FORCEINLINE constexpr FwdIter loop_with_cleanup_n(
        Iter it, std::size_t count, FwdIter dest, F&& f, Cleanup&& cleanup)
    {
        using cat = typename std::iterator_traits<Iter>::iterator_category;
        return detail::loop_with_cleanup_n<cat>::call(
            it, count, dest, PIKA_FORWARD(F, f), PIKA_FORWARD(Cleanup, cleanup));
    }

    template <typename Iter, typename CancelToken, typename F, typename Cleanup>
    PIKA_FORCEINLINE constexpr Iter loop_with_cleanup_n_with_token(
        Iter it, std::size_t count, CancelToken& tok, F&& f, Cleanup&& cleanup)
    {
        using cat = typename std::iterator_traits<Iter>::iterator_category;
        return detail::loop_with_cleanup_n<cat>::call_with_token(
            it, count, tok, PIKA_FORWARD(F, f), PIKA_FORWARD(Cleanup, cleanup));
    }

    template <typename Iter, typename FwdIter, typename CancelToken, typename F,
        typename Cleanup>
    PIKA_FORCEINLINE constexpr FwdIter loop_with_cleanup_n_with_token(Iter it,
        std::size_t count, FwdIter dest, CancelToken& tok, F&& f,
        Cleanup&& cleanup)
    {
        using cat = typename std::iterator_traits<Iter>::iterator_category;
        return detail::loop_with_cleanup_n<cat>::call_with_token(it, count,
            dest, tok, PIKA_FORWARD(F, f), PIKA_FORWARD(Cleanup, cleanup));
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // Helper class to repeatedly call a function a given number of times
        // starting from a given iterator position.
        template <typename IterCat>
        struct loop_idx_n
        {
            ///////////////////////////////////////////////////////////////////
            // handle sequences of non-futures
            template <typename Iter, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr Iter call(
                std::size_t base_idx, Iter it, std::size_t num, F&& f)
            {
                std::size_t count(num & std::size_t(-4));    // -V112

                for (std::size_t i = 0; i < count;
                     (void) ++it, i += 4)    // -V112
                {
                    PIKA_INVOKE(f, *it, base_idx++);
                    PIKA_INVOKE(f, *++it, base_idx++);
                    PIKA_INVOKE(f, *++it, base_idx++);
                    PIKA_INVOKE(f, *++it, base_idx++);
                }
                for (/**/; count < num; (void) ++count, ++it, ++base_idx)
                {
                    PIKA_INVOKE(f, *it, base_idx);
                }
                return it;
            }

            template <typename Iter, typename CancelToken, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr Iter call(
                std::size_t base_idx, Iter it, std::size_t count,
                CancelToken& tok, F&& f)
            {
                for (/**/; count != 0; (void) --count, ++it, ++base_idx)
                {
                    if (tok.was_cancelled(base_idx))
                    {
                        break;
                    }
                    PIKA_INVOKE(f, *it, base_idx);
                }
                return it;
            }
        };

        template <>
        struct loop_idx_n<std::random_access_iterator_tag>
        {
            ///////////////////////////////////////////////////////////////////
            // handle sequences of non-futures
            template <typename Iter, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr Iter call(
                std::size_t base_idx, Iter it, std::size_t num, F&& f)
            {
                while (num >= 4)
                {
                    PIKA_INVOKE(f, *it, base_idx++);
                    PIKA_INVOKE(f, *(it + 1), base_idx++);
                    PIKA_INVOKE(f, *(it + 2), base_idx++);
                    PIKA_INVOKE(f, *(it + 3), base_idx++);

                    it += 4;
                    num -= 4;
                }

                switch (num)
                {
                case 3:
                    PIKA_INVOKE(f, *it, base_idx++);
                    PIKA_INVOKE(f, *(it + 1), base_idx++);
                    PIKA_INVOKE(f, *(it + 2), base_idx++);
                    break;

                case 2:
                    PIKA_INVOKE(f, *it, base_idx++);
                    PIKA_INVOKE(f, *(it + 1), base_idx++);
                    break;

                case 1:
                    PIKA_INVOKE(f, *it, base_idx);
                    break;

                default:
                    break;
                }

                return it + num;
            }

            template <typename Iter, typename CancelToken, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static constexpr Iter call(
                std::size_t base_idx, Iter it, std::size_t num,
                CancelToken& tok, F&& f)
            {
                while (num >= 4)
                {
                    if (tok.was_cancelled(base_idx))
                        return it;

                    PIKA_INVOKE(f, *it, base_idx++);
                    PIKA_INVOKE(f, *(it + 1), base_idx++);
                    PIKA_INVOKE(f, *(it + 2), base_idx++);
                    PIKA_INVOKE(f, *(it + 3), base_idx++);

                    it += 4;
                    num -= 4;
                }

                switch (num)
                {
                case 3:
                    PIKA_INVOKE(f, *it, base_idx++);
                    PIKA_INVOKE(f, *(it + 1), base_idx++);
                    PIKA_INVOKE(f, *(it + 2), base_idx++);
                    break;

                case 2:
                    PIKA_INVOKE(f, *it, base_idx++);
                    PIKA_INVOKE(f, *(it + 1), base_idx++);
                    break;

                case 1:
                    PIKA_INVOKE(f, *it, base_idx);
                    break;

                default:
                    break;
                }

                return it + num;
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct loop_idx_n_t final
      : pika::functional::detail::tag_fallback<loop_idx_n_t<ExPolicy>>
    {
    private:
        template <typename Iter, typename F>
        friend PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr Iter
        tag_fallback_invoke(pika::parallel::util::loop_idx_n_t<ExPolicy>,
            std::size_t base_idx, Iter it, std::size_t count, F&& f)
        {
            using cat = typename std::iterator_traits<Iter>::iterator_category;
            return detail::loop_idx_n<cat>::call(
                base_idx, it, count, PIKA_FORWARD(F, f));
        }

        template <typename Iter, typename CancelToken, typename F>
        friend PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr Iter
        tag_fallback_invoke(pika::parallel::util::loop_idx_n_t<ExPolicy>,
            std::size_t base_idx, Iter it, std::size_t count, CancelToken& tok,
            F&& f)
        {
            using cat = typename std::iterator_traits<Iter>::iterator_category;
            return detail::loop_idx_n<cat>::call(
                base_idx, it, count, tok, PIKA_FORWARD(F, f));
        }
    };

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr loop_idx_n_t<ExPolicy> loop_idx_n =
        loop_idx_n_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename Iter, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr Iter loop_idx_n(
        std::size_t base_idx, Iter it, std::size_t count, F&& f)
    {
        return pika::parallel::util::loop_idx_n_t<ExPolicy>{}(
            base_idx, it, count, PIKA_FORWARD(F, f));
    }

    template <typename ExPolicy, typename Iter, typename CancelToken,
        typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr Iter loop_idx_n(
        std::size_t base_idx, Iter it, std::size_t count, CancelToken& tok,
        F&& f)
    {
        return pika::parallel::util::loop_idx_n_t<ExPolicy>{}(
            base_idx, it, count, tok, PIKA_FORWARD(F, f));
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        // Helper class to repeatedly call a function a given number of times
        // starting from a given iterator position.
        template <typename IterCat>
        struct accumulate_n
        {
            template <typename Iter, typename T, typename Pred>
            static T call(Iter it, std::size_t count, T init, Pred&& f)
            {
                for (/**/; count != 0; (void) --count, ++it)
                {
                    init = PIKA_INVOKE(f, init, *it);
                }
                return init;
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename T, typename Pred>
    PIKA_FORCEINLINE T accumulate_n(Iter it, std::size_t count, T init, Pred&& f)
    {
        using cat = typename std::iterator_traits<Iter>::iterator_category;
        return detail::accumulate_n<cat>::call(
            it, count, PIKA_MOVE(init), PIKA_FORWARD(Pred, f));
    }

    template <typename T, typename Iter, typename Reduce,
        typename Conv = util::projection_identity>
    PIKA_FORCEINLINE T accumulate(
        Iter first, Iter last, Reduce&& r, Conv&& conv = Conv())
    {
        T val = PIKA_INVOKE(conv, *first);
        ++first;
        while (last != first)
        {
            val = PIKA_INVOKE(r, val, *first);
            ++first;
        }
        return val;
    }

    template <typename T, typename Iter1, typename Iter2, typename Reduce,
        typename Conv>
    PIKA_FORCEINLINE T accumulate(
        Iter1 first1, Iter1 last1, Iter2 first2, Reduce&& r, Conv&& conv)
    {
        T val = PIKA_INVOKE(conv, *first1, *first2);
        ++first1;
        ++first2;
        while (last1 != first1)
        {
            val = PIKA_INVOKE(r, val, PIKA_INVOKE(conv, *first1, *first2));
            ++first1;
            ++first2;
        }
        return val;
    }
}}}    // namespace pika::parallel::util
