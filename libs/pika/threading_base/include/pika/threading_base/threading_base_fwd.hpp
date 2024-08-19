//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file pika/runtime/threads/thread_data_fwd.hpp

#pragma once

#include <pika/config.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/coroutines/coroutine_fwd.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/coroutines/thread_id_type.hpp>
#include <pika/functional/function.hpp>
#include <pika/functional/unique_function.hpp>
#include <pika/modules/errors.hpp>
#endif

#if defined(PIKA_HAVE_APEX)
# include <apex_api.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

namespace pika::detail::external_timer {
#if defined(PIKA_HAVE_APEX)
    using apex::task_wrapper;
#else
    struct task_wrapper
    {
    };
#endif
}    // namespace pika::detail::external_timer

namespace pika::threads {
    class PIKA_EXPORT thread_pool_base;
}    // namespace pika::threads

namespace pika::threads::detail {

    /// \cond NOINTERNAL
    struct scheduler_base;
    class thread_data;
    class thread_data_stackful;
    class thread_data_stackless;

    using thread_id_ref_type = thread_id_ref;
    using thread_id_type = thread_id;

    using coroutine_type = coroutines::detail::coroutine;
    using stackless_coroutine_type = coroutines::detail::stackless_coroutine;

    using thread_result_type = std::pair<thread_schedule_state, thread_id_type>;
    using thread_arg_type = thread_restart_state;

    using thread_function_sig = thread_result_type(thread_arg_type);
    using thread_function_type = util::detail::unique_function<thread_function_sig>;

    using thread_self = coroutines::detail::coroutine_self;
    using thread_self_impl_type = coroutines::detail::coroutine_impl;

#if defined(PIKA_HAVE_APEX)
    PIKA_EXPORT std::shared_ptr<pika::detail::external_timer::task_wrapper> get_self_timer_data(
        void);
    PIKA_EXPORT void set_self_timer_data(
        std::shared_ptr<pika::detail::external_timer::task_wrapper> data);
#endif
    /// \endcond
}    // namespace pika::threads::detail
