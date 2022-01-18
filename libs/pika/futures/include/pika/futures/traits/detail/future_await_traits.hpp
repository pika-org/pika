//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

#if defined(PIKA_HAVE_CXX20_COROUTINES)

#include <pika/futures/detail/future_data.hpp>
#include <pika/futures/traits/future_access.hpp>
#include <pika/modules/allocator_support.hpp>
#include <pika/modules/memory.hpp>

#include <coroutine>
#include <cstddef>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace lcos { namespace detail {

    template <typename Promise = void>
    using coroutine_handle = std::coroutine_handle<Promise>;
    using suspend_never = std::suspend_never;

    ///////////////////////////////////////////////////////////////////////////
    // this was removed from the TS, so we define our own
    struct suspend_if
    {
        bool is_ready_;

        constexpr explicit suspend_if(bool cond) noexcept
          : is_ready_(!cond)
        {
        }

        PIKA_NODISCARD constexpr bool await_ready() const noexcept
        {
            return is_ready_;
        }
        constexpr void await_suspend(coroutine_handle<>) const noexcept {}
        constexpr void await_resume() const noexcept {}
    };

    ///////////////////////////////////////////////////////////////////////////
    // Allow using co_await with an expression which evaluates to
    // pika::future<T>.
    template <typename T>
    PIKA_FORCEINLINE bool await_ready(pika::future<T> const& f) noexcept
    {
        return f.is_ready();
    }

    template <typename T, typename Promise>
    PIKA_FORCEINLINE void await_suspend(
        pika::future<T>& f, coroutine_handle<Promise> rh)
    {
        // f.then([=](future<T> result) {});
        auto st = traits::detail::get_shared_state(f);
        st->set_on_completed([st, rh]() mutable {
            if (st->has_exception())
            {
                rh.promise().set_exception(st->get_exception_ptr());
            }
            rh();
        });
    }

    template <typename T>
    PIKA_FORCEINLINE T await_resume(pika::future<T>& f)
    {
        return f.get();
    }

    // Allow wrapped futures to be unwrapped, if possible.
    template <typename T>
    PIKA_FORCEINLINE T await_resume(pika::future<pika::future<T>>& f)
    {
        return f.get().get();
    }

    template <typename T>
    PIKA_FORCEINLINE T await_resume(pika::future<pika::shared_future<T>>& f)
    {
        return f.get().get();
    }

    // Allow using co_await with an expression which evaluates to
    // pika::shared_future<T>.
    template <typename T>
    PIKA_FORCEINLINE bool await_ready(pika::shared_future<T> const& f) noexcept
    {
        return f.is_ready();
    }

    template <typename T, typename Promise>
    PIKA_FORCEINLINE void await_suspend(
        pika::shared_future<T>& f, coroutine_handle<Promise> rh)
    {
        // f.then([=](shared_future<T> result) {})
        auto st = traits::detail::get_shared_state(f);
        st->set_on_completed([st, rh]() mutable {
            if (st->has_exception())
            {
                rh.promise().set_exception(st->get_exception_ptr());
            }
            rh();
        });
    }

    template <typename T>
    PIKA_FORCEINLINE T await_resume(pika::shared_future<T>& f)
    {
        return f.get();
    }

    ///////////////////////////////////////////////////////////////////////////
    // derive from future shared state as this will be combined with the
    // necessary stack frame for the resumable function
    template <typename T, typename Derived>
    struct coroutine_promise_base : pika::lcos::detail::future_data<T>
    {
        using base_type = pika::lcos::detail::future_data<T>;
        using init_no_addref = typename base_type::init_no_addref;

        using allocator_type = pika::util::internal_allocator<char>;

        // the shared state is held alive by the coroutine
        coroutine_promise_base()
          : base_type(init_no_addref{})
        {
        }

        pika::future<T> get_return_object()
        {
            pika::intrusive_ptr<Derived> shared_state(
                static_cast<Derived*>(this));
            return pika::traits::future_access<pika::future<T>>::create(
                PIKA_MOVE(shared_state));
        }

        constexpr suspend_never initial_suspend() const noexcept
        {
            return suspend_never{};
        }

        suspend_if final_suspend() noexcept
        {
            // This gives up the coroutine's reference count on the shared
            // state. If this was the last reference count, the coroutine
            // should not suspend before exiting.
            return suspend_if{!this->base_type::requires_delete()};
        }

        void destroy() noexcept override
        {
            coroutine_handle<Derived>::from_promise(
                *static_cast<Derived*>(this))
                .destroy();
        }

        // allocator support for shared coroutine state
        PIKA_NODISCARD static void* allocate(std::size_t size)
        {
            using char_allocator = typename std::allocator_traits<
                allocator_type>::template rebind_alloc<char>;
            using traits = std::allocator_traits<char_allocator>;
            using unique_ptr = std::unique_ptr<char,
                pika::util::allocator_deleter<char_allocator>>;

            char_allocator alloc{};
            unique_ptr p(traits::allocate(alloc, size),
                pika::util::allocator_deleter<char_allocator>{alloc});

            return p.release();
        }

        static void deallocate(void* p, std::size_t size) noexcept
        {
            using char_allocator = typename std::allocator_traits<
                allocator_type>::template rebind_alloc<char>;
            using traits = std::allocator_traits<char_allocator>;

            char_allocator alloc{};
            traits::deallocate(alloc, static_cast<char*>(p), size);
        }
    };
}}}    // namespace pika::lcos::detail

///////////////////////////////////////////////////////////////////////////////
namespace std {
    // Allow for functions which use co_await to return an pika::future<T>
    template <typename T, typename... Ts>
    struct coroutine_traits<pika::future<T>, Ts...>
    {
        using allocator_type = pika::util::internal_allocator<coroutine_traits>;

        struct promise_type
          : pika::lcos::detail::coroutine_promise_base<T, promise_type>
        {
            using base_type =
                pika::lcos::detail::coroutine_promise_base<T, promise_type>;

            promise_type() = default;

            template <typename U>
            void return_value(U&& value)
            {
                this->base_type::set_value(PIKA_FORWARD(U, value));
            }

            void unhandled_exception() noexcept
            {
                this->base_type::set_exception(std::current_exception());
            }

            PIKA_NODISCARD PIKA_FORCEINLINE static void* operator new(
                std::size_t size)
            {
                return base_type::allocate(size);
            }

            PIKA_FORCEINLINE static void operator delete(
                void* p, std::size_t size) noexcept
            {
                base_type::deallocate(p, size);
            }
        };
    };

    template <typename... Ts>
    struct coroutine_traits<pika::future<void>, Ts...>
    {
        using allocator_type = pika::util::internal_allocator<coroutine_traits>;

        struct promise_type
          : pika::lcos::detail::coroutine_promise_base<void, promise_type>
        {
            using base_type =
                pika::lcos::detail::coroutine_promise_base<void, promise_type>;

            promise_type() = default;

            void return_void()
            {
                this->base_type::set_value();
            }

            void unhandled_exception() noexcept
            {
                this->base_type::set_exception(std::current_exception());
            }

            PIKA_NODISCARD PIKA_FORCEINLINE static void* operator new(
                std::size_t size)
            {
                return base_type::allocate(size);
            }

            PIKA_FORCEINLINE static void operator delete(
                void* p, std::size_t size) noexcept
            {
                base_type::deallocate(p, size);
            }
        };
    };

    // Allow for functions which use co_await to return an
    // pika::shared_future<T>
    template <typename T, typename... Ts>
    struct coroutine_traits<pika::shared_future<T>, Ts...>
    {
        using allocator_type = pika::util::internal_allocator<coroutine_traits>;

        struct promise_type
          : pika::lcos::detail::coroutine_promise_base<T, promise_type>
        {
            using base_type =
                pika::lcos::detail::coroutine_promise_base<T, promise_type>;

            promise_type() = default;

            template <typename U>
            void return_value(U&& value)
            {
                this->base_type::set_value(PIKA_FORWARD(U, value));
            }

            void unhandled_exception() noexcept
            {
                this->base_type::set_exception(std::current_exception());
            }

            PIKA_NODISCARD PIKA_FORCEINLINE static void* operator new(
                std::size_t size)
            {
                return base_type::allocate(size);
            }

            PIKA_FORCEINLINE static void operator delete(
                void* p, std::size_t size) noexcept
            {
                base_type::deallocate(p, size);
            }
        };
    };

    template <typename... Ts>
    struct coroutine_traits<pika::shared_future<void>, Ts...>
    {
        using allocator_type = pika::util::internal_allocator<coroutine_traits>;

        struct promise_type
          : pika::lcos::detail::coroutine_promise_base<void, promise_type>
        {
            using base_type =
                pika::lcos::detail::coroutine_promise_base<void, promise_type>;

            promise_type() = default;

            void return_void()
            {
                this->base_type::set_value();
            }

            void unhandled_exception() noexcept
            {
                this->base_type::set_exception(std::current_exception());
            }

            PIKA_NODISCARD PIKA_FORCEINLINE static void* operator new(
                std::size_t size)
            {
                return base_type::allocate(size);
            }

            PIKA_FORCEINLINE static void operator delete(
                void* p, std::size_t size) noexcept
            {
                base_type::deallocate(p, size);
            }
        };
    };
}    // namespace std

#endif    // PIKA_HAVE_CXX20_COROUTINES
