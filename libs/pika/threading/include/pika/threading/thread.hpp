//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/functional/deferred_call.hpp>
#include <pika/functional/function.hpp>
#include <pika/functional/unique_function.hpp>
#include <pika/futures/future_fwd.hpp>
#include <pika/modules/errors.hpp>
#include <pika/synchronization/spinlock.hpp>
#include <pika/threading_base/scheduler_base.hpp>
#include <pika/threading_base/thread_data.hpp>
#include <pika/threading_base/thread_pool_base.hpp>
#include <pika/timing/steady_clock.hpp>

#include <cstddef>
#include <exception>
#include <functional>
#include <iosfwd>
#include <mutex>
#include <type_traits>
#include <utility>

#include <pika/local/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace pika {
    ///////////////////////////////////////////////////////////////////////////
    using thread_termination_handler_type =
        util::function_nonser<void(std::exception_ptr const& e)>;
    PIKA_EXPORT void set_thread_termination_handler(
        thread_termination_handler_type f);

    class PIKA_EXPORT thread
    {
        typedef lcos::local::spinlock mutex_type;
        void terminate(const char* function, const char* reason) const;

    public:
        class id;
        typedef threads::thread_id_type native_handle_type;

        thread() noexcept;

        template <typename F,
            typename Enable = typename std::enable_if<!std::is_same<
                typename std::decay<F>::type, thread>::value>::type>
        explicit thread(F&& f)
        {
            auto thrd_data = threads::get_self_id_data();
            PIKA_ASSERT(thrd_data);
            start_thread(thrd_data->get_scheduler_base()->get_parent_pool(),
                util::deferred_call(PIKA_FORWARD(F, f)));
        }

        template <typename F, typename... Ts>
        explicit thread(F&& f, Ts&&... vs)
        {
            auto thrd_data = threads::get_self_id_data();
            PIKA_ASSERT(thrd_data);
            start_thread(thrd_data->get_scheduler_base()->get_parent_pool(),
                util::deferred_call(PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, vs)...));
        }

        template <typename F>
        thread(threads::thread_pool_base* pool, F&& f)
        {
            start_thread(pool, util::deferred_call(PIKA_FORWARD(F, f)));
        }

        template <typename F, typename... Ts>
        thread(threads::thread_pool_base* pool, F&& f, Ts&&... vs)
        {
            start_thread(pool,
                util::deferred_call(PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, vs)...));
        }

        ~thread();

    public:
        thread(thread&&) noexcept;
        thread& operator=(thread&&) noexcept;

        void swap(thread&) noexcept;
        bool joinable() const noexcept
        {
            std::lock_guard<mutex_type> l(mtx_);
            return joinable_locked();
        }

        void join();
        void detach()
        {
            std::lock_guard<mutex_type> l(mtx_);
            detach_locked();
        }

        id get_id() const noexcept;

        native_handle_type native_handle() const    //-V659
        {
            std::lock_guard<mutex_type> l(mtx_);
            return id_.noref();
        }

        PIKA_NODISCARD static unsigned int hardware_concurrency() noexcept;

        // extensions
        void interrupt(bool flag = true);
        bool interruption_requested() const;

        static void interrupt(id, bool flag = true);

        pika::future<void> get_future(error_code& ec = throws);

        std::size_t get_thread_data() const;
        std::size_t set_thread_data(std::size_t);

#if defined(PIKA_HAVE_LIBCDS)
        std::size_t get_libcds_data() const;
        std::size_t set_libcds_data(std::size_t);
        std::size_t get_libcds_hazard_pointer_data() const;
        std::size_t set_libcds_hazard_pointer_data(std::size_t);
        std::size_t get_libcds_dynamic_hazard_pointer_data() const;
        std::size_t set_libcds_dynamic_hazard_pointer_data(std::size_t);
#endif

    private:
        bool joinable_locked() const noexcept
        {
            return threads::invalid_thread_id != id_;
        }
        void detach_locked()
        {
            id_ = threads::invalid_thread_id;
        }
        void start_thread(threads::thread_pool_base* pool,
            util::unique_function_nonser<void()>&& func);
        static threads::thread_result_type thread_function_nullary(
            util::unique_function_nonser<void()> const& func);

        mutable mutex_type mtx_;
        threads::thread_id_ref_type id_;
    };

    inline void swap(thread& x, thread& y) noexcept
    {
        x.swap(y);
    }

    ///////////////////////////////////////////////////////////////////////////
    class thread::id
    {
    private:
        threads::thread_id_type id_;

        friend bool operator==(
            thread::id const& x, thread::id const& y) noexcept;
        friend bool operator!=(
            thread::id const& x, thread::id const& y) noexcept;
        friend bool operator<(
            thread::id const& x, thread::id const& y) noexcept;
        friend bool operator>(
            thread::id const& x, thread::id const& y) noexcept;
        friend bool operator<=(
            thread::id const& x, thread::id const& y) noexcept;
        friend bool operator>=(
            thread::id const& x, thread::id const& y) noexcept;

        template <typename Char, typename Traits>
        friend std::basic_ostream<Char, Traits>& operator<<(
            std::basic_ostream<Char, Traits>&, thread::id const&);

        friend class thread;

    public:
        id() noexcept = default;

        explicit id(threads::thread_id_type const& i) noexcept
          : id_(i)
        {
        }
        explicit id(threads::thread_id_type&& i) noexcept
          : id_(PIKA_MOVE(i))
        {
        }

        explicit id(threads::thread_id_ref_type const& i) noexcept
          : id_(i.get().get())
        {
        }
        explicit id(threads::thread_id_ref_type&& i) noexcept
          : id_(PIKA_MOVE(i).get().get())
        {
        }

        threads::thread_id_type const& native_handle() const
        {
            return id_;
        }
    };

    inline bool operator==(thread::id const& x, thread::id const& y) noexcept
    {
        return x.id_ == y.id_;
    }

    inline bool operator!=(thread::id const& x, thread::id const& y) noexcept
    {
        return !(x == y);
    }

    inline bool operator<(thread::id const& x, thread::id const& y) noexcept
    {
        return x.id_ < y.id_;
    }

    inline bool operator>(thread::id const& x, thread::id const& y) noexcept
    {
        return y < x;
    }

    inline bool operator<=(thread::id const& x, thread::id const& y) noexcept
    {
        return !(x > y);
    }

    inline bool operator>=(thread::id const& x, thread::id const& y) noexcept
    {
        return !(x < y);
    }

    template <typename Char, typename Traits>
    std::basic_ostream<Char, Traits>& operator<<(
        std::basic_ostream<Char, Traits>& out, thread::id const& id)
    {
        out << id.id_;
        return out;
    }

    //     template <class T> struct hash;
    //     template <> struct hash<thread::id>;

    ///////////////////////////////////////////////////////////////////////////
    namespace this_thread {
        PIKA_EXPORT thread::id get_id() noexcept;

        PIKA_EXPORT void yield() noexcept;
        PIKA_EXPORT void yield_to(thread::id) noexcept;

        // extensions
        PIKA_EXPORT threads::thread_priority get_priority();
        PIKA_EXPORT std::ptrdiff_t get_stack_size();

        PIKA_EXPORT void interruption_point();
        PIKA_EXPORT bool interruption_enabled();
        PIKA_EXPORT bool interruption_requested();

        PIKA_EXPORT void interrupt();

        PIKA_EXPORT void sleep_until(
            pika::chrono::steady_time_point const& abs_time);

        inline void sleep_for(pika::chrono::steady_duration const& rel_time)
        {
            sleep_until(rel_time.from_now());
        }

        PIKA_EXPORT std::size_t get_thread_data();
        PIKA_EXPORT std::size_t set_thread_data(std::size_t);

#if defined(PIKA_HAVE_LIBCDS)
        PIKA_EXPORT std::size_t get_libcds_data();
        PIKA_EXPORT std::size_t set_libcds_data(std::size_t);
        PIKA_EXPORT std::size_t get_libcds_hazard_pointer_data();
        PIKA_EXPORT std::size_t set_libcds_hazard_pointer_data(
            std::size_t);
        PIKA_EXPORT std::size_t get_libcds_dynamic_hazard_pointer_data();
        PIKA_EXPORT std::size_t set_libcds_dynamic_hazard_pointer_data(
            std::size_t);
#endif

        class PIKA_EXPORT disable_interruption
        {
        private:
            disable_interruption(disable_interruption const&);
            disable_interruption& operator=(disable_interruption const&);

            bool interruption_was_enabled_;
            friend class restore_interruption;

        public:
            disable_interruption();
            ~disable_interruption();
        };

        class PIKA_EXPORT restore_interruption
        {
        private:
            restore_interruption(restore_interruption const&);
            restore_interruption& operator=(restore_interruption const&);

            bool interruption_was_enabled_;

        public:
            explicit restore_interruption(disable_interruption& d);
            ~restore_interruption();
        };
    }    // namespace this_thread
}    // namespace pika

namespace std {

    // specialize std::hash for pika::thread::id
    template <>
    struct hash<::pika::thread::id>
    {
        std::size_t operator()(::pika::thread::id const& id) const
        {
            std::hash<::pika::threads::thread_id_ref_type> hasher_;
            return hasher_(id.native_handle());
        }
    };
}    // namespace std

#include <pika/local/config/warnings_suffix.hpp>
