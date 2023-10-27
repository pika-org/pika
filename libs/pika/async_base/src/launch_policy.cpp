//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/async_base/launch_policy.hpp>
#include <pika/coroutines/thread_enums.hpp>

namespace pika {
    ///////////////////////////////////////////////////////////////////////////
    const detail::async_policy launch::async =
        detail::async_policy{execution::thread_priority::default_};
    const detail::fork_policy launch::fork =
        detail::fork_policy{execution::thread_priority::default_};
    const detail::sync_policy launch::sync = detail::sync_policy{};
    const detail::deferred_policy launch::deferred = detail::deferred_policy{};
    const detail::apply_policy launch::apply = detail::apply_policy{};

    const detail::select_policy_generator launch::select = detail::select_policy_generator{};

    const detail::policy_holder<> launch::all = detail::policy_holder<>{detail::launch_policy::all};
    const detail::policy_holder<> launch::sync_policies =
        detail::policy_holder<>{detail::launch_policy::sync_policies};
    const detail::policy_holder<> launch::async_policies =
        detail::policy_holder<>{detail::launch_policy::async_policies};
}    // namespace pika
