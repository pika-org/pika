//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_STDEXEC)
# include <pika/execution_base/stdexec_forward.hpp>
#endif

#include <pika/allocator_support/allocator_deleter.hpp>
#include <pika/allocator_support/internal_allocator.hpp>
#include <pika/allocator_support/traits/is_allocator.hpp>
#include <pika/errors/try_catch_exception_ptr.hpp>
#include <pika/execution/algorithms/detail/helpers.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution_base/operation_state.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/futures/detail/future_data.hpp>
#include <pika/futures/promise.hpp>
#include <pika/memory/intrusive_ptr.hpp>
#include <pika/type_support/detail/with_result_of.hpp>
#include <pika/type_support/pack.hpp>
#include <pika/type_support/unused.hpp>

#include <exception>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

namespace pika::make_future_detail {
    template <typename T, typename Allocator>
    struct resettable_operation_state_future_data
      : pika::lcos::detail::future_data_allocator<T, Allocator>
    {
        PIKA_NON_COPYABLE(resettable_operation_state_future_data);

        virtual void reset_operation_state() = 0;

        using init_no_addref =
            typename pika::lcos::detail::future_data_allocator<T, Allocator>::init_no_addref;

        template <typename Allocator_>
        resettable_operation_state_future_data(init_no_addref no_addref, Allocator_ const& alloc)
          : pika::lcos::detail::future_data_allocator<T, Allocator>(no_addref, alloc)
        {
        }
    };

    template <typename T, typename Allocator>
    struct make_future_receiver
    {
        using is_receiver = void;

        pika::intrusive_ptr<resettable_operation_state_future_data<T, Allocator>> data;

        friend void tag_invoke(pika::execution::experimental::set_error_t, make_future_receiver&& r,
            std::exception_ptr ep) noexcept
        {
            // We move the receiver into a local variable from the operation
            // state which the receiver refers to. This allows us to safely
            // reset the operation state without destroying the receiver.
            make_future_receiver r_local = PIKA_MOVE(r);

            r_local.data->set_exception(PIKA_MOVE(ep));
            r_local.data->reset_operation_state();
            r_local.data.reset();
        }

        friend void tag_invoke(
            pika::execution::experimental::set_stopped_t, make_future_receiver&&) noexcept
        {
            std::terminate();
        }

        template <typename U>
        friend void tag_invoke(
            pika::execution::experimental::set_value_t, make_future_receiver&& r, U&& u) noexcept
        {
            // We move the receiver into a local variable from the operation
            // state which the receiver refers to. This allows us to safely
            // reset the operation state without destroying the receiver.
            make_future_receiver r_local = PIKA_MOVE(r);

            pika::detail::try_catch_exception_ptr(
                [&]() { r_local.data->set_value(PIKA_FORWARD(U, u)); },
                [&](std::exception_ptr ep) { r_local.data->set_exception(PIKA_MOVE(ep)); });
            r_local.data->reset_operation_state();
            r_local.data.reset();
        }

        friend constexpr pika::execution::experimental::empty_env tag_invoke(
            pika::execution::experimental::get_env_t, make_future_receiver const&) noexcept
        {
            return {};
        }
    };

    template <typename Allocator>
    struct make_future_receiver<void, Allocator>
    {
        pika::intrusive_ptr<resettable_operation_state_future_data<void, Allocator>> data;

        friend void tag_invoke(pika::execution::experimental::set_error_t, make_future_receiver&& r,
            std::exception_ptr ep) noexcept
        {
            // We move the receiver into a local variable from the operation
            // state which the receiver refers to. This allows us to safely
            // reset the operation state without destroying the receiver.
            make_future_receiver r_local = PIKA_MOVE(r);

            r_local.data->set_exception(PIKA_MOVE(ep));
            r_local.data->reset_operation_state();
            r_local.data.reset();
        }

        friend void tag_invoke(
            pika::execution::experimental::set_stopped_t, make_future_receiver&&) noexcept
        {
            std::terminate();
        }

        friend void tag_invoke(
            pika::execution::experimental::set_value_t, make_future_receiver&& r) noexcept
        {
            // We move the receiver into a local variable from the operation
            // state which the receiver refers to. This allows us to safely
            // reset the operation state without destroying the receiver.
            make_future_receiver r_local = PIKA_MOVE(r);

            pika::detail::try_catch_exception_ptr(
                [&]() { r_local.data->set_value(pika::util::detail::unused); },
                [&](std::exception_ptr ep) { r_local.data->set_exception(PIKA_MOVE(ep)); });
            r_local.data->reset_operation_state();
            r_local.data.reset();
        }

        friend constexpr pika::execution::experimental::empty_env tag_invoke(
            pika::execution::experimental::get_env_t, make_future_receiver const&) noexcept
        {
            return {};
        }
    };

    template <typename T, typename Allocator, typename OperationState>
    struct future_data : resettable_operation_state_future_data<T, Allocator>
    {
        PIKA_NON_COPYABLE(future_data);

        using operation_state_type = std::decay_t<OperationState>;
        using other_allocator =
            typename std::allocator_traits<Allocator>::template rebind_alloc<future_data>;
        using init_no_addref =
            typename pika::lcos::detail::future_data_allocator<T, Allocator>::init_no_addref;

        // The operation state is stored in an optional so that it can be
        // reset explicitly as soon as set_* is called.
        std::optional<operation_state_type> op_state;

        template <typename Sender>
        future_data(init_no_addref no_addref, other_allocator const& alloc, Sender&& sender)
          : resettable_operation_state_future_data<T, Allocator>(no_addref, alloc)
        {
            op_state.emplace(pika::detail::with_result_of([&]() {
                return pika::execution::experimental::connect(
                    PIKA_FORWARD(Sender, sender), make_future_receiver<T, Allocator>{this});
            }));
            pika::execution::experimental::start(op_state.value());
        }

        void reset_operation_state() override
        {
            op_state.reset();
        }
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename Sender, typename Allocator>
    auto make_future(Sender&& sender, Allocator const& allocator)
    {
        using allocator_type = Allocator;

#if defined(PIKA_HAVE_STDEXEC)
        using value_types =
            typename pika::execution::experimental::value_types_of_t<std::decay_t<Sender>,
                pika::execution::experimental::empty_env, pika::util::detail::pack,
                pika::util::detail::pack>;
#else
        using value_types =
            typename pika::execution::experimental::sender_traits<std::decay_t<Sender>>::
                template value_types<pika::util::detail::pack, pika::util::detail::pack>;
#endif
        using result_type =
            std::decay_t<pika::execution::experimental::detail::single_result_t<value_types>>;
        using operation_state_type = std::invoke_result_t<pika::execution::experimental::connect_t,
            Sender, make_future_receiver<result_type, allocator_type>>;

        using shared_state = future_data<result_type, allocator_type, operation_state_type>;
        using init_no_addref = typename shared_state::init_no_addref;
        using other_allocator =
            typename std::allocator_traits<allocator_type>::template rebind_alloc<shared_state>;
        using allocator_traits = std::allocator_traits<other_allocator>;
        using unique_ptr =
            std::unique_ptr<shared_state, pika::detail::allocator_deleter<other_allocator>>;

        other_allocator alloc(allocator);
        unique_ptr p(allocator_traits::allocate(alloc, 1),
            pika::detail::allocator_deleter<other_allocator>{alloc});

        allocator_traits::construct(
            alloc, p.get(), init_no_addref{}, alloc, PIKA_FORWARD(Sender, sender));

        return pika::traits::future_access<future<result_type>>::create(p.release(), false);
    }
}    // namespace pika::make_future_detail

namespace pika::execution::experimental {
    inline constexpr struct make_future_t final
      : pika::functional::detail::tag_fallback<make_future_t>
    {
    private:
        // clang-format off
        template <typename Sender,
            typename Allocator = pika::detail::internal_allocator<>,
            PIKA_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                pika::detail::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            make_future_t, Sender&& sender, Allocator const& allocator = Allocator{})
        {
            return make_future_detail::make_future(PIKA_FORWARD(Sender, sender), allocator);
        }

        // clang-format off
        template <typename Allocator = pika::detail::internal_allocator<>,
            PIKA_CONCEPT_REQUIRES_(
                pika::detail::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(make_future_t, Allocator const& allocator = Allocator{})
        {
            return detail::partial_algorithm<make_future_t, Allocator>{allocator};
        }
    } make_future{};
}    // namespace pika::execution::experimental
