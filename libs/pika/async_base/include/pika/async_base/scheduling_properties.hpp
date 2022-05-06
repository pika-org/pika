//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/tag_invoke.hpp>

namespace pika::execution::experimental {
    inline constexpr struct with_priority_t final
      : pika::functional::tag<with_priority_t>
    {
    } with_priority{};

    inline constexpr struct get_priority_t final
      : pika::functional::tag<get_priority_t>
    {
    } get_priority{};

    inline constexpr struct with_stacksize_t final
      : pika::functional::tag<with_stacksize_t>
    {
    } with_stacksize{};

    inline constexpr struct get_stacksize_t final
      : pika::functional::tag<get_stacksize_t>
    {
    } get_stacksize{};

    inline constexpr struct with_hint_t final
      : pika::functional::tag<with_hint_t>
    {
    } with_hint{};

    inline constexpr struct get_hint_t final : pika::functional::tag<get_hint_t>
    {
    } get_hint{};

    // with_annotation uses tag_fallback as the base class to allow an
    // out-of-line fallback implementation for executors that don't support
    // annotations by themselves. See annotating_executor.
    inline constexpr struct with_annotation_t final
      : pika::functional::detail::tag_fallback<with_annotation_t>
    {
    } with_annotation{};

    inline constexpr struct get_annotation_t final
      : pika::functional::tag<get_annotation_t>
    {
    } get_annotation{};
}    // namespace pika::execution::experimental
