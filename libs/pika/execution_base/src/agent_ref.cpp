//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/assert.hpp>
#include <pika/execution_base/agent_ref.hpp>
#ifdef PIKA_HAVE_VERIFY_LOCKS
# include <pika/lock_registration/detail/register_locks.hpp>
#endif
#include <pika/execution_base/this_thread.hpp>

#include <fmt/format.h>

#include <cstddef>
#include <string>

namespace pika::execution::detail {

    void agent_ref::yield(const char* desc)
    {
        PIKA_ASSERT(*this == pika::execution::this_thread::detail::agent());
        // verify that there are no more registered locks for this OS-thread
#ifdef PIKA_HAVE_VERIFY_LOCKS
        util::verify_no_locks();
#endif
        impl_->yield(desc);
    }

    void agent_ref::yield_k(std::size_t k, const char* desc)
    {
        PIKA_ASSERT(*this == pika::execution::this_thread::detail::agent());
        // verify that there are no more registered locks for this OS-thread
#ifdef PIKA_HAVE_VERIFY_LOCKS
        util::verify_no_locks();
#endif
        impl_->yield_k(k, desc);
    }

    void agent_ref::spin_k(std::size_t k, const char* desc)
    {
        PIKA_ASSERT(*this == pika::execution::this_thread::detail::agent());
        // verify that there are no more registered locks for this OS-thread
#ifdef PIKA_HAVE_VERIFY_LOCKS
        util::verify_no_locks();
#endif
        impl_->spin_k(k, desc);
    }

    void agent_ref::suspend(const char* desc)
    {
        PIKA_ASSERT(*this == pika::execution::this_thread::detail::agent());
        // verify that there are no more registered locks for this OS-thread
#ifdef PIKA_HAVE_VERIFY_LOCKS
        util::verify_no_locks();
#endif
        impl_->suspend(desc);
    }

    void agent_ref::resume(const char* desc)
    {
        PIKA_ASSERT(*this != pika::execution::this_thread::detail::agent());
        impl_->resume(desc);
    }

    void agent_ref::abort(const char* desc)
    {
        PIKA_ASSERT(*this != pika::execution::this_thread::detail::agent());
        impl_->abort(desc);
    }

    void agent_ref::sleep_for(pika::chrono::steady_duration const& sleep_duration, const char* desc)
    {
        PIKA_ASSERT(*this == pika::execution::this_thread::detail::agent());
        impl_->sleep_for(sleep_duration, desc);
    }

    void agent_ref::sleep_until(pika::chrono::steady_time_point const& sleep_time, const char* desc)
    {
        PIKA_ASSERT(*this == pika::execution::this_thread::detail::agent());
        impl_->sleep_until(sleep_time, desc);
    }

    std::string format(pika::execution::detail::agent_ref const& a)
    {
        return fmt::format("agent_ref{{{}}}", a.impl_->description());
    }
}    // namespace pika::execution::detail
