//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/futures/traits/future_traits.hpp>
#include <pika/modules/memory.hpp>
#include <pika/type_support/unused.hpp>

#include <type_traits>
#include <utility>
#include <vector>

namespace pika {
    template <typename R>
    class future;
    template <typename R>
    class shared_future;

    namespace lcos::detail {
        template <typename Result>
        struct future_data_base;
    }    // namespace lcos::detail
}    // namespace pika

namespace pika { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        struct future_data_void
        {
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        struct shared_state_ptr_result
        {
            using type = Result;
        };

        template <typename Result>
        struct shared_state_ptr_result<Result&>
        {
            using type = Result&;
        };

        template <>
        struct shared_state_ptr_result<void>
        {
            using type = future_data_void;
        };

        template <typename Future>
        using shared_state_ptr_result_t =
            typename shared_state_ptr_result<Future>::type;

        ///////////////////////////////////////////////////////////////////////
        template <typename R>
        struct shared_state_ptr
        {
            using result_type = shared_state_ptr_result_t<R>;
            using type =
                pika::intrusive_ptr<lcos::detail::future_data_base<result_type>>;
        };

        template <typename Future>
        using shared_state_ptr_t = typename shared_state_ptr<Future>::type;

        ///////////////////////////////////////////////////////////////////////
        template <typename Future, typename Enable = void>
        struct shared_state_ptr_for
          : shared_state_ptr<typename traits::future_traits<Future>::type>
        {
        };

        template <typename Future>
        struct shared_state_ptr_for<Future const> : shared_state_ptr_for<Future>
        {
        };

        template <typename Future>
        struct shared_state_ptr_for<Future&> : shared_state_ptr_for<Future>
        {
        };

        template <typename Future>
        struct shared_state_ptr_for<Future&&> : shared_state_ptr_for<Future>
        {
        };

        template <typename Future>
        struct shared_state_ptr_for<std::vector<Future>>
        {
            using type =
                std::vector<typename shared_state_ptr_for<Future>::type>;
        };

        template <typename Future>
        using shared_state_ptr_for_t =
            typename shared_state_ptr_for<Future>::type;

        ///////////////////////////////////////////////////////////////////////
        template <typename SharedState, typename Allocator>
        struct shared_state_allocator;
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct is_shared_state : std::false_type
    {
    };

    template <typename R>
    struct is_shared_state<
        pika::intrusive_ptr<lcos::detail::future_data_base<R>>> : std::true_type
    {
    };

    template <typename R>
    inline constexpr bool is_shared_state_v = is_shared_state<R>::value;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename T, typename Enable = void>
        struct future_access_customization_point;
    }

    template <typename T>
    struct future_access : detail::future_access_customization_point<T>
    {
    };

    template <typename R>
    struct future_access<pika::future<R>>
    {
        template <typename SharedState>
        static pika::future<R> create(
            pika::intrusive_ptr<SharedState> const& shared_state)
        {
            return pika::future<R>(shared_state);
        }

        template <typename T = void>
        static pika::future<R> create(
            detail::shared_state_ptr_for_t<pika::future<pika::future<R>>> const&
                shared_state)
        {
            return pika::future<pika::future<R>>(shared_state);
        }

        template <typename SharedState>
        static pika::future<R> create(
            pika::intrusive_ptr<SharedState>&& shared_state)
        {
            return pika::future<R>(PIKA_MOVE(shared_state));
        }

        template <typename T = void>
        static pika::future<R> create(
            detail::shared_state_ptr_for_t<pika::future<pika::future<R>>>&&
                shared_state)
        {
            return pika::future<pika::future<R>>(PIKA_MOVE(shared_state));
        }

        template <typename SharedState>
        static pika::future<R> create(
            SharedState* shared_state, bool addref = true)
        {
            return pika::future<R>(
                pika::intrusive_ptr<SharedState>(shared_state, addref));
        }

        PIKA_FORCEINLINE static traits::detail::shared_state_ptr_t<R> const&
        get_shared_state(pika::future<R> const& f)
        {
            return f.shared_state_;
        }

        PIKA_FORCEINLINE static
            typename traits::detail::shared_state_ptr_t<R>::element_type*
            detach_shared_state(pika::future<R>&& f)
        {
            return f.shared_state_.detach();
        }

    private:
        template <typename Destination>
        PIKA_FORCEINLINE static void transfer_result_impl(
            pika::future<R>&& src, Destination& dest, std::false_type)
        {
            dest.set_value(src.get());
        }

        template <typename Destination>
        PIKA_FORCEINLINE static void transfer_result_impl(
            pika::future<R>&& src, Destination& dest, std::true_type)
        {
            src.get();
            dest.set_value(util::unused);
        }

    public:
        template <typename Destination>
        PIKA_FORCEINLINE static void transfer_result(
            pika::future<R>&& src, Destination& dest)
        {
            transfer_result_impl(PIKA_MOVE(src), dest, std::is_void<R>{});
        }
    };

    template <typename R>
    struct future_access<pika::shared_future<R>>
    {
        template <typename SharedState>
        static pika::shared_future<R> create(
            pika::intrusive_ptr<SharedState> const& shared_state)
        {
            return pika::shared_future<R>(shared_state);
        }

        template <typename T = void>
        static pika::shared_future<R> create(detail::shared_state_ptr_for_t<
            pika::shared_future<pika::future<R>>> const& shared_state)
        {
            return pika::shared_future<pika::future<R>>(shared_state);
        }

        template <typename SharedState>
        static pika::shared_future<R> create(
            pika::intrusive_ptr<SharedState>&& shared_state)
        {
            return pika::shared_future<R>(PIKA_MOVE(shared_state));
        }

        template <typename T = void>
        static pika::shared_future<R> create(
            detail::shared_state_ptr_for_t<pika::shared_future<pika::future<R>>>&&
                shared_state)
        {
            return pika::shared_future<pika::future<R>>(PIKA_MOVE(shared_state));
        }

        template <typename SharedState>
        static pika::shared_future<R> create(
            SharedState* shared_state, bool addref = true)
        {
            return pika::shared_future<R>(
                pika::intrusive_ptr<SharedState>(shared_state, addref));
        }

        PIKA_FORCEINLINE static traits::detail::shared_state_ptr_t<R> const&
        get_shared_state(pika::shared_future<R> const& f)
        {
            return f.shared_state_;
        }

        PIKA_FORCEINLINE static
            typename traits::detail::shared_state_ptr_t<R>::element_type*
            detach_shared_state(pika::shared_future<R> const& f)
        {
            return f.shared_state_.get();
        }

    private:
        template <typename Destination>
        PIKA_FORCEINLINE static void transfer_result_impl(
            pika::shared_future<R>&& src, Destination& dest, std::false_type)
        {
            dest.set_value(src.get());
        }

        template <typename Destination>
        PIKA_FORCEINLINE static void transfer_result_impl(
            pika::shared_future<R>&& src, Destination& dest, std::true_type)
        {
            src.get();
            dest.set_value(util::unused);
        }

    public:
        template <typename Destination>
        PIKA_FORCEINLINE static void transfer_result(
            pika::shared_future<R>&& src, Destination& dest)
        {
            transfer_result_impl(PIKA_MOVE(src), dest, std::is_void<R>{});
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename SharedState, typename Allocator>
    struct shared_state_allocator
      : detail::shared_state_allocator<SharedState, Allocator>
    {
    };

    template <typename SharedState, typename Allocator>
    using shared_state_allocator_t =
        typename shared_state_allocator<SharedState, Allocator>::type;
}}    // namespace pika::traits
