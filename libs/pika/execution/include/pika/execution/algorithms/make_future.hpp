//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/allocator_support/allocator_deleter.hpp>
#include <pika/allocator_support/internal_allocator.hpp>
#include <pika/allocator_support/traits/is_allocator.hpp>
#include <pika/errors/try_catch_exception_ptr.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution/algorithms/detail/single_result.hpp>
#include <pika/execution_base/operation_state.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/invoke_result.hpp>
#include <pika/futures/detail/future_data.hpp>
#include <pika/futures/promise.hpp>
#include <pika/modules/memory.hpp>
#include <pika/type_support/pack.hpp>
#include <pika/type_support/unused.hpp>

#include <exception>
#include <memory>
#include <utility>

namespace pika { namespace execution { namespace experimental {
    namespace detail {
        template <typename T, typename Allocator>
        struct make_future_receiver
        {
            pika::intrusive_ptr<
                pika::lcos::detail::future_data_allocator<T, Allocator>>
                data;

            friend void tag_invoke(set_error_t, make_future_receiver&& r,
                std::exception_ptr ep) noexcept
            {
                r.data->set_exception(PIKA_MOVE(ep));
                r.data.reset();
            }

            friend void tag_invoke(set_done_t, make_future_receiver&&) noexcept
            {
                std::terminate();
            }

            template <typename U>
            friend void tag_invoke(
                set_value_t, make_future_receiver&& r, U&& u) noexcept
            {
                pika::detail::try_catch_exception_ptr(
                    [&]() { r.data->set_value(PIKA_FORWARD(U, u)); },
                    [&](std::exception_ptr ep) {
                        r.data->set_exception(PIKA_MOVE(ep));
                    });
                r.data.reset();
            }
        };

        template <typename Allocator>
        struct make_future_receiver<void, Allocator>
        {
            pika::intrusive_ptr<
                pika::lcos::detail::future_data_allocator<void, Allocator>>
                data;

            friend void tag_invoke(set_error_t, make_future_receiver&& r,
                std::exception_ptr ep) noexcept
            {
                r.data->set_exception(PIKA_MOVE(ep));
                r.data.reset();
            }

            friend void tag_invoke(set_done_t, make_future_receiver&&) noexcept
            {
                std::terminate();
            }

            friend void tag_invoke(
                set_value_t, make_future_receiver&& r) noexcept
            {
                pika::detail::try_catch_exception_ptr(
                    [&]() { r.data->set_value(pika::util::unused); },
                    [&](std::exception_ptr ep) {
                        r.data->set_exception(PIKA_MOVE(ep));
                    });
                r.data.reset();
            }
        };

        template <typename T, typename Allocator, typename OperationState>
        struct future_data
          : pika::lcos::detail::future_data_allocator<T, Allocator>
        {
            PIKA_NON_COPYABLE(future_data);

            using operation_state_type = std::decay_t<OperationState>;
            using init_no_addref =
                typename pika::lcos::detail::future_data_allocator<T,
                    Allocator>::init_no_addref;
            using other_allocator = typename std::allocator_traits<
                Allocator>::template rebind_alloc<future_data>;

            operation_state_type op_state;

            template <typename Sender>
            future_data(init_no_addref no_addref, other_allocator const& alloc,
                Sender&& sender)
              : pika::lcos::detail::future_data_allocator<T, Allocator>(
                    no_addref, alloc)
              , op_state(pika::execution::experimental::connect(
                    PIKA_FORWARD(Sender, sender),
                    detail::make_future_receiver<T, Allocator>{this}))
            {
                pika::execution::experimental::start(op_state);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Sender, typename Allocator>
        auto make_future(Sender&& sender, Allocator const& allocator)
        {
            using allocator_type = Allocator;

            using value_types =
                typename pika::execution::experimental::sender_traits<
                    std::decay_t<Sender>>::template value_types<pika::util::pack,
                    pika::util::pack>;
            using result_type =
                std::decay_t<detail::single_result_t<value_types>>;
            using operation_state_type = pika::util::invoke_result_t<
                pika::execution::experimental::connect_t, Sender,
                detail::make_future_receiver<result_type, allocator_type>>;

            using shared_state = detail::future_data<result_type,
                allocator_type, operation_state_type>;
            using init_no_addref = typename shared_state::init_no_addref;
            using other_allocator = typename std::allocator_traits<
                allocator_type>::template rebind_alloc<shared_state>;
            using allocator_traits = std::allocator_traits<other_allocator>;
            using unique_ptr = std::unique_ptr<shared_state,
                util::allocator_deleter<other_allocator>>;

            other_allocator alloc(allocator);
            unique_ptr p(allocator_traits::allocate(alloc, 1),
                pika::util::allocator_deleter<other_allocator>{alloc});

            allocator_traits::construct(alloc, p.get(), init_no_addref{}, alloc,
                PIKA_FORWARD(Sender, sender));

            return pika::traits::future_access<future<result_type>>::create(
                p.release(), false);
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    inline constexpr struct make_future_t final
      : pika::functional::detail::tag_fallback<make_future_t>
    {
    private:
        // clang-format off
        template <typename Sender,
            typename Allocator = pika::util::internal_allocator<>,
            PIKA_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                pika::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(make_future_t,
            Sender&& sender, Allocator const& allocator = Allocator{})
        {
            return detail::make_future(PIKA_FORWARD(Sender, sender), allocator);
        }

        // clang-format off
        template <typename Allocator = pika::util::internal_allocator<>,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            make_future_t, Allocator const& allocator = Allocator{})
        {
            return detail::partial_algorithm<make_future_t, Allocator>{
                allocator};
        }
    } make_future{};
}}}    // namespace pika::execution::experimental
