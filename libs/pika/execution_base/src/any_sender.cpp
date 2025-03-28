//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/errors/error.hpp>
#include <pika/errors/throw_exception.hpp>
#include <pika/execution_base/any_sender.hpp>

#include <fmt/format.h>

#include <atomic>
#include <exception>
#include <string>
#include <utility>

namespace pika::execution::experimental::detail {
    void empty_any_operation_state_holder_state::start() & noexcept
    {
        PIKA_THROW_EXCEPTION(pika::error::bad_function_call, "any_operation_state::start",
            "attempted to call start on empty any_operation_state");
    }

    bool any_operation_state_holder_base::empty() const noexcept { return false; }
    bool empty_any_operation_state_holder_state::empty() const noexcept { return true; }
    void any_operation_state_holder::start() & noexcept { storage.get().start(); }

    void throw_bad_any_call(char const* class_name, char const* function_name)
    {
        PIKA_THROW_EXCEPTION(pika::error::bad_function_call,
            fmt::format("{}::{}", class_name, function_name), "attempted to call {} on empty {}",
            function_name, class_name);
    }
}    // namespace pika::execution::experimental::detail
