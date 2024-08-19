//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/coroutines/detail/coroutine_impl.hpp>
#include <pika/coroutines/detail/coroutine_stackful_self.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/coroutines/thread_id_type.hpp>
#include <pika/execution_base/agent_base.hpp>
#include <pika/execution_base/context_base.hpp>
#include <pika/execution_base/resource_base.hpp>
#include <pika/timing/steady_clock.hpp>
#endif

#include <cstddef>
#include <string>

#include <pika/config/warnings_prefix.hpp>

namespace pika::threads::detail {
    struct PIKA_EXPORT execution_context : pika::execution::detail::context_base
    {
        pika::execution::detail::resource_base const& resource() const override
        {
            return resource_;
        }
        pika::execution::detail::resource_base resource_;
    };

    struct PIKA_EXPORT execution_agent : pika::execution::detail::agent_base
    {
        explicit execution_agent(coroutines::detail::coroutine_impl* coroutine) noexcept;

        std::string description() const override;

        execution_context const& context() const override { return context_; }

        void yield(char const* desc) override;
        void yield_k(std::size_t k, char const* desc) override;
        void spin_k(std::size_t k, char const* desc) override;
        void suspend(char const* desc) override;
        void resume(char const* desc) override;
        void abort(char const* desc) override;
        void sleep_for(
            pika::chrono::steady_duration const& sleep_duration, char const* desc) override;
        void sleep_until(
            pika::chrono::steady_time_point const& sleep_time, char const* desc) override;

    private:
        coroutines::detail::coroutine_stackful_self self_;

        thread_restart_state do_yield(char const* desc, thread_schedule_state state);

        void do_resume(char const* desc, thread_restart_state statex);

        execution_context context_;
    };
}    // namespace pika::threads::detail

#include <pika/config/warnings_suffix.hpp>
