//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/errors/try_catch_exception_ptr.hpp>
#include <pika/functional/traits/is_invocable.hpp>
#include <pika/functional/unique_function.hpp>
#include <pika/futures/detail/future_data.hpp>
#include <pika/futures/promise.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/futures.hpp>
#include <pika/threading_base/annotated_function.hpp>

#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

namespace pika { namespace lcos { namespace local {

    template <typename Sig>
    class packaged_task;

    template <typename R, typename... Ts>
    class packaged_task<R(Ts...)>
    {
        using function_type = util::unique_function_nonser<R(Ts...)>;

    public:
        // construction and destruction
        packaged_task() = default;

        template <typename F, typename FD = std::decay_t<F>,
            typename Enable =
                std::enable_if_t<!std::is_same_v<FD, packaged_task> &&
                    is_invocable_r_v<R, FD&, Ts...>>>
        explicit packaged_task(F&& f)
          : function_(PIKA_FORWARD(F, f))
          , promise_()
        {
        }

        template <typename Allocator, typename F, typename FD = std::decay_t<F>,
            typename Enable =
                std::enable_if_t<!std::is_same_v<FD, packaged_task> &&
                    is_invocable_r_v<R, FD&, Ts...>>>
        explicit packaged_task(std::allocator_arg_t, Allocator const& a, F&& f)
          : function_(PIKA_FORWARD(F, f))
          , promise_(std::allocator_arg, a)
        {
        }

        packaged_task(packaged_task const& rhs) noexcept = delete;
        packaged_task(packaged_task&& rhs) noexcept = default;

        packaged_task& operator=(packaged_task const& rhs) noexcept = delete;
        packaged_task& operator=(packaged_task&& rhs) noexcept = default;

        void swap(packaged_task& rhs) noexcept
        {
            function_.swap(rhs.function_);
            promise_.swap(rhs.promise_);
        }

        void operator()(Ts... ts)
        {
            if (function_.empty())
            {
                PIKA_THROW_EXCEPTION(no_state,
                    "packaged_task<Signature>::operator()",
                    "this packaged_task has no valid shared state");
                return;
            }

            // synchronous execution of the embedded function (object)
            pika::detail::try_catch_exception_ptr(
                [&]() {
                    pika::scoped_annotation annotate(function_);
                    if constexpr (std::is_void_v<R>)
                    {
                        function_(PIKA_FORWARD(Ts, ts)...);
                        promise_.set_value();
                    }
                    else
                    {
                        promise_.set_value(function_(PIKA_FORWARD(Ts, ts)...));
                    }
                },
                [&](std::exception_ptr ep) {
                    promise_.set_exception(PIKA_MOVE(ep));
                });
        }

        // result retrieval
        pika::future<R> get_future(error_code& ec = throws)
        {
            if (function_.empty())
            {
                PIKA_THROWS_IF(ec, no_state,
                    "packaged_task<Signature>::get_future",
                    "this packaged_task has no valid shared state");
                return pika::future<R>();
            }
            return promise_.get_future();
        }

        bool valid() const noexcept
        {
            return !function_.empty() && promise_.valid();
        }

        void reset(error_code& ec = throws)
        {
            if (function_.empty())
            {
                PIKA_THROWS_IF(ec, no_state, "packaged_task<Signature>::reset",
                    "this packaged_task has no valid shared state");
                return;
            }
            promise_ = local::promise<R>();
        }

        // extension
        void set_exception(std::exception_ptr const& e)
        {
            promise_.set_exception(e);
        }

    private:
        function_type function_;
        local::promise<R> promise_;
    };
}}}    // namespace pika::lcos::local

namespace std {
    // Requires: Allocator shall be an allocator (17.6.3.5)
    template <typename Sig, typename Allocator>
    struct uses_allocator<pika::lcos::local::packaged_task<Sig>, Allocator>
      : std::true_type
    {
    };

    template <typename Sig>
    void swap(pika::lcos::local::packaged_task<Sig>& lhs,
        pika::lcos::local::packaged_task<Sig>& rhs)
    {
        lhs.swap(rhs);
    }
}    // namespace std
