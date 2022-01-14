//  Copyright (c) 2016-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/async_base/traits/is_launch_policy.hpp>
#include <pika/execution/traits/executor_traits.hpp>
#include <pika/execution_base/execution.hpp>

#include <type_traits>
#include <utility>

namespace pika { namespace parallel { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        /// \cond NOINTERNAL
        template <typename Category1, typename Category2>
        struct is_not_weaker : std::false_type
        {
        };

        template <typename Category>
        struct is_not_weaker<Category, Category> : std::true_type
        {
        };

        template <>
        struct is_not_weaker<pika::execution::parallel_execution_tag,
            pika::execution::unsequenced_execution_tag> : std::true_type
        {
        };

        template <>
        struct is_not_weaker<pika::execution::sequenced_execution_tag,
            pika::execution::unsequenced_execution_tag> : std::true_type
        {
        };

        template <>
        struct is_not_weaker<pika::execution::sequenced_execution_tag,
            pika::execution::parallel_execution_tag> : std::true_type
        {
        };
        /// \endcond
    }    // namespace detail

    /// Rebind the type of executor used by an execution policy. The execution
    /// category of Executor shall not be weaker than that of ExecutionPolicy.
    template <typename ExPolicy, typename Executor, typename Parameters>
    struct rebind_executor
    {
        /// \cond NOINTERNAL
        using policy_type = std::decay_t<ExPolicy>;
        using executor_type = std::decay_t<Executor>;
        using parameters_type = std::decay_t<Parameters>;

        using category1 = typename policy_type::execution_category;
        using category2 = typename pika::traits::executor_execution_category<
            executor_type>::type;

        static_assert(detail::is_not_weaker<category2, category1>::value,
            "detail::is_not_weaker<category2, category1>::value");
        /// \endcond

        /// The type of the rebound execution policy
        using type = typename policy_type::template rebind<executor_type,
            parameters_type>::type;
    };

    //////////////////////////////////////////////////////////////////////////
    struct create_rebound_policy_t
    {
        template <typename ExPolicy, typename Executor, typename Parameters>
        constexpr decltype(auto) operator()(
            ExPolicy&&, Executor&& exec, Parameters&& parameters) const
        {
            using rebound_type =
                typename rebind_executor<ExPolicy, Executor, Parameters>::type;

            return rebound_type(PIKA_FORWARD(Executor, exec),
                PIKA_FORWARD(Parameters, parameters));
        }
    };

    inline constexpr create_rebound_policy_t create_rebound_policy{};
}}}    // namespace pika::parallel::execution
