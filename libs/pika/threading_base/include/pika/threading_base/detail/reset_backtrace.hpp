//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#ifdef PIKA_HAVE_THREAD_BACKTRACE_ON_SUSPENSION

#include <pika/errors/error_code.hpp>
#include <pika/threading_base/threading_base_fwd.hpp>

#include <memory>
#include <string>

namespace pika::threads::detail {
    struct reset_backtrace
    {
        PIKA_EXPORT explicit reset_backtrace(
            thread_id_type const& id, error_code& ec = throws);
        PIKA_EXPORT ~reset_backtrace();

        thread_id_type id_;
        std::unique_ptr<pika::util::backtrace> backtrace_;
#ifdef PIKA_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
        std::string full_backtrace_;
#endif
        error_code& ec_;
    };
}    // namespace pika::threads::detail

#endif    // PIKA_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
