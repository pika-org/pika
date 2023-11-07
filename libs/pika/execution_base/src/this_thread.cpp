//  Copyright (c) 2013-2019 Thomas Heller
//  Copyright (c) 2008 Peter Dimov
//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/errors/throw_exception.hpp>
#include <pika/execution_base/agent_base.hpp>
#include <pika/execution_base/context_base.hpp>
#include <pika/execution_base/detail/spinlock_deadlock_detection.hpp>
#include <pika/execution_base/this_thread.hpp>
#include <pika/timing/steady_clock.hpp>

#ifdef PIKA_HAVE_SPINLOCK_DEADLOCK_DETECTION
# include <pika/errors/throw_exception.hpp>
#endif

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>
#include <fmt/std.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#if defined(PIKA_WINDOWS)
# include <windows.h>
#else
# ifndef _AIX
#  include <sched.h>
# else
// AIX's sched.h defines ::var which sometimes conflicts with Lambda's var
extern "C" int sched_yield(void);
# endif
# include <time.h>
#endif

namespace pika::execution {

    namespace detail { namespace {

        struct default_context : context_base
        {
            resource_base const& resource() const override { return resource_; }
            resource_base resource_;
        };

        struct default_agent : detail::agent_base
        {
            default_agent();

            std::string description() const override { return fmt::format("{}", id_); }

            default_context const& context() const override { return context_; }

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
            bool running_;
            bool aborted_;
            std::thread::id id_;
            std::mutex mtx_;
            std::condition_variable suspend_cv_;
            std::condition_variable resume_cv_;

            default_context context_;
        };

        default_agent::default_agent()
          : running_(true)
          , aborted_(false)
          , id_(std::this_thread::get_id())
        {
        }

        void default_agent::yield(char const* /* desc */)
        {
#if defined(PIKA_WINDOWS)
            Sleep(0);
#else
            sched_yield();
#endif
        }

        void default_agent::yield_k(std::size_t k, char const* /* desc */)
        {
            if (k < 4) {}
            else if (k < 16) { PIKA_SMT_PAUSE; }
            else if (k < 32 || k & 1)
            {
#if defined(PIKA_WINDOWS)
                Sleep(0);
#else
                sched_yield();
#endif
            }
            else
            {
#if defined(PIKA_WINDOWS)
                Sleep(1);
#else
                // g++ -Wextra warns on {} or {0}
                struct timespec rqtp = {0, 0};

                // POSIX says that timespec has tv_sec and tv_nsec
                // But it doesn't guarantee order or placement

                rqtp.tv_sec = 0;
                rqtp.tv_nsec = 1000;

                nanosleep(&rqtp, nullptr);
#endif
            }
        }

        void default_agent::spin_k(std::size_t k, char const* /* desc */)
        {
            if (k < 4) {}
            else { PIKA_SMT_PAUSE; }
        }

        void default_agent::suspend(char const* /* desc */)
        {
            std::unique_lock<std::mutex> l(mtx_);
            PIKA_ASSERT(running_);

            running_ = false;
            resume_cv_.notify_all();

            while (!running_) { suspend_cv_.wait(l); }

            if (aborted_)
            {
                PIKA_THROW_EXCEPTION(pika::error::yield_aborted, "suspend",
                    "std::thread({}) aborted (yield returned wait_abort)", id_);
            }
        }

        void default_agent::resume(char const* /* desc */)
        {
            {
                std::unique_lock<std::mutex> l(mtx_);
                while (running_) { resume_cv_.wait(l); }
                running_ = true;
            }
            suspend_cv_.notify_one();
        }

        void default_agent::abort(char const* /* desc */)
        {
            {
                std::unique_lock<std::mutex> l(mtx_);
                while (running_) { resume_cv_.wait(l); }
                running_ = true;
                aborted_ = true;
            }
            suspend_cv_.notify_one();
        }

        void default_agent::sleep_for(
            pika::chrono::steady_duration const& sleep_duration, char const* /* desc */)
        {
            std::this_thread::sleep_for(sleep_duration.value());
        }

        void default_agent::sleep_until(
            pika::chrono::steady_time_point const& sleep_time, char const* /* desc */)
        {
            std::this_thread::sleep_until(sleep_time.value());
        }
    }}    // namespace detail

    namespace detail {
        agent_base& get_default_agent()
        {
            static thread_local default_agent agent;
            return agent;
        }
    }    // namespace detail

    namespace this_thread::detail {

        struct agent_storage
        {
            agent_storage()
              : impl_(&pika::execution::detail::get_default_agent())
            {
            }

            execution::detail::agent_base* set(execution::detail::agent_base* context) noexcept
            {
                std::swap(context, impl_);
                return context;
            }

            execution::detail::agent_base* impl_;
        };

        agent_storage* get_agent_storage()
        {
            static thread_local agent_storage storage;
            return &storage;
        }

        reset_agent::reset_agent(
            detail::agent_storage* storage, execution::detail::agent_base& impl)
          : storage_(storage)
          , old_(storage_->set(&impl))
        {
        }

        reset_agent::reset_agent(execution::detail::agent_base& impl)
          : reset_agent(detail::get_agent_storage(), impl)
        {
        }

        reset_agent::~reset_agent() { storage_->set(old_); }

        pika::execution::detail::agent_ref agent()
        {
            return pika::execution::detail::agent_ref(detail::get_agent_storage()->impl_);
        }

        void yield(char const* desc) { agent().yield(desc); }

        void yield_k(std::size_t k, char const* desc)
        {
#ifdef PIKA_HAVE_SPINLOCK_DEADLOCK_DETECTION
            if (pika::util::detail::get_spinlock_break_on_deadlock_enabled())
            {
                if (desc == nullptr) { desc = ""; }

                auto const deadlock_detection_limit =
                    pika::util::detail::get_spinlock_deadlock_detection_limit();
                if (k >= deadlock_detection_limit)
                {
                    PIKA_THROW_EXCEPTION(pika::error::deadlock, desc,
                        "yield_k yielded {} times. This may indicate a deadlock in your "
                        "application or a bug in pika. Stopping because "
                        "pika.spinlock_deadlock_detection_limit={}.",
                        k, deadlock_detection_limit);
                }

                auto const deadlock_warning_limit =
                    pika::util::detail::get_spinlock_deadlock_warning_limit();
                if (k >= deadlock_warning_limit && k % deadlock_warning_limit == 0)
                {
                    fmt::print(std::cerr,
                        "desc: {}. yield_k already yielded {} times "
                        "(pika.spinlock_deadlock_warning_limit={}). This may indicate a deadlock "
                        "in your application or a bug in pika. Stopping after "
                        "pika.spinlock_deadlock_detection_limit={} iterations.\n",
                        desc, k, deadlock_warning_limit, deadlock_detection_limit);
                }
            }

#endif

            agent().yield_k(k, desc);
        }

        void spin_k(std::size_t k, char const* desc)
        {
#ifdef PIKA_HAVE_SPINLOCK_DEADLOCK_DETECTION
            if (pika::util::detail::get_spinlock_break_on_deadlock_enabled())
            {
                if (desc == nullptr) { desc = ""; }

                auto const deadlock_detection_limit =
                    pika::util::detail::get_spinlock_deadlock_detection_limit();
                if (k >= deadlock_detection_limit)
                {
                    PIKA_THROW_EXCEPTION(pika::error::deadlock, desc,
                        "spin_k spun {} times. This may indicate a deadlock in your "
                        "application or a bug in pika. Stopping because "
                        "pika.spinlock_deadlock_detection_limit={}.",
                        k, deadlock_detection_limit);
                }

                auto const deadlock_warning_limit =
                    pika::util::detail::get_spinlock_deadlock_warning_limit();
                if (k >= deadlock_warning_limit && k % deadlock_warning_limit == 0)
                {
                    fmt::print(std::cerr,
                        "desc: {}. spin_k spun {} times "
                        "(pika.spinlock_deadlock_warning_limit={}). This may indicate a deadlock "
                        "in your application or a bug in pika. Stopping after "
                        "pika.spinlock_deadlock_detection_limit={} iterations.\n",
                        desc, k, deadlock_warning_limit, deadlock_detection_limit);
                }
            }

#endif

            agent().spin_k(k, desc);
        }

        void suspend(char const* desc) { agent().suspend(desc); }
    }    // namespace this_thread::detail
}    // namespace pika::execution
