//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/functional/bind.hpp>
#include <pika/functional/bind_back.hpp>
#include <pika/functional/bind_front.hpp>
#include <pika/functional/function.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/functional/invoke_fused.hpp>
#include <pika/functional/mem_fn.hpp>
#include <pika/functional/traits/is_bind_expression.hpp>
#include <pika/functional/traits/is_placeholder.hpp>
#include <pika/functional/unique_function.hpp>
#include <pika/threading_base/annotated_function.hpp>
#include <pika/threading_base/scoped_annotation.hpp>

namespace pika {
    using pika::traits::is_bind_expression;
    using pika::traits::is_placeholder;
    using pika::util::bind_back;
    using pika::util::bind_front;
    using pika::util::function;
    using pika::util::function_nonser;
    using pika::util::invoke;
    using pika::util::invoke_fused;
    using pika::util::mem_fn;
    using pika::util::unique_function;
    using pika::util::unique_function_nonser;

    namespace placeholders {
        using namespace pika::util::placeholders;
    }
}    // namespace pika
