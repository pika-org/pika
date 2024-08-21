//  Copyright (c)      2022 ETH Zurich
//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/threading_base/thread_description.hpp>
#include <pika/threading_base/threading_base_fwd.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/coroutines/thread_id_type.hpp>

#include <cstdint>
#include <memory>
#endif

namespace pika::detail::external_timer {
    inline std::shared_ptr<task_wrapper> new_task(
        pika::detail::thread_description const&, std::uint32_t, threads::detail::thread_id_type)
    {
        return nullptr;
    }

    inline std::shared_ptr<task_wrapper> update_task(
        std::shared_ptr<task_wrapper>, pika::detail::thread_description const&)
    {
        return nullptr;
    }

    struct [[nodiscard]] scoped_timer
    {
        explicit scoped_timer(std::shared_ptr<task_wrapper>){};
        scoped_timer(scoped_timer&&) = delete;
        scoped_timer(scoped_timer const&) = delete;
        scoped_timer& operator=(scoped_timer&&) = delete;
        scoped_timer& operator=(scoped_timer const&) = delete;
        ~scoped_timer() = default;

        void stop() {}
        void yield() {}
    };
}    // namespace pika::detail::external_timer
