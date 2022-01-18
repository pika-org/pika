//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file pika/execution/execution_policy_annotation.hpp

#pragma once

#include <pika/local/config.hpp>
#include <pika/execution/executors/execution_parameters.hpp>
#include <pika/execution/executors/rebind_executor.hpp>
#include <pika/execution/traits/is_execution_policy.hpp>
#include <pika/executors/annotating_executor.hpp>
#include <pika/properties/property.hpp>

#include <string>
#include <type_traits>
#include <utility>

namespace pika { namespace execution { namespace experimental {

    // with_annotation property implementation for execution policies
    // that simply forwards to the embedded executor
    // clang-format off
    template <typename ExPolicy,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy_v<ExPolicy>
        )>
    // clang-format on
    constexpr decltype(auto) tag_invoke(
        pika::execution::experimental::with_annotation_t, ExPolicy&& policy,
        char const* annotation)
    {
        auto exec = pika::execution::experimental::with_annotation(
            policy.executor(), annotation);

        return pika::parallel::execution::create_rebound_policy(
            policy, PIKA_MOVE(exec), policy.parameters());
    }

    // clang-format off
    template <typename ExPolicy,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy_v<ExPolicy>
        )>
    // clang-format on
    decltype(auto) tag_invoke(pika::execution::experimental::with_annotation_t,
        ExPolicy&& policy, std::string annotation)
    {
        auto exec = pika::execution::experimental::with_annotation(
            policy.executor(), PIKA_MOVE(annotation));

        return pika::parallel::execution::create_rebound_policy(
            policy, PIKA_MOVE(exec), policy.parameters());
    }

    // get_annotation property implementation for execution policies
    // that simply forwards to the embedded executor
    // clang-format off
    template <typename ExPolicy,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy_v<ExPolicy> &&
            pika::functional::is_tag_invocable_v<
                pika::execution::experimental::get_annotation_t,
                typename std::decay_t<ExPolicy>::executor_type>
        )>
    // clang-format on
    constexpr decltype(auto) tag_invoke(
        pika::execution::experimental::get_annotation_t, ExPolicy&& policy)
    {
        return pika::execution::experimental::get_annotation(policy.executor());
    }
}}}    // namespace pika::execution::experimental
