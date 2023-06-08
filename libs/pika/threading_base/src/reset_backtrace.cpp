//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>

#ifdef PIKA_HAVE_THREAD_BACKTRACE_ON_SUSPENSION

# include <pika/modules/debugging.hpp>
# include <pika/modules/errors.hpp>
# include <pika/threading_base/thread_helpers.hpp>
# include <pika/threading_base/threading_base_fwd.hpp>

namespace pika::threads::detail {
    reset_backtrace::reset_backtrace(thread_id_type const& id, error_code& ec)
      : id_(id)
      , backtrace_(new pika::debug::detail::backtrace())
# ifdef PIKA_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
      , full_backtrace_(backtrace_->trace())
# endif
      , ec_(ec)
    {
# ifdef PIKA_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
        threads::detail::set_thread_backtrace(id_, full_backtrace_.c_str(), ec_);
# else
        threads::detail::set_thread_backtrace(id_, backtrace_.get(), ec_);
# endif
    }
    reset_backtrace::~reset_backtrace()
    {
        threads::detail::set_thread_backtrace(id_, 0, ec_);
    }
}    // namespace pika::threads::detail

#endif    // PIKA_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
