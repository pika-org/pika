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
#include <pika/assert.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/execution_base/operation_state.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/type_support/unused.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

namespace pika::start_detached_detail {
    template <typename Sender, typename Allocator>
    struct operation_state_holder
    {
        struct start_detached_receiver
        {
            PIKA_STDEXEC_RECEIVER_CONCEPT

            operation_state_holder& op_state;

            template <typename Error>
#if !defined(__NVCC__)
            [[noreturn]]
#endif
            void set_error(Error&& error) && noexcept
            {
                auto r = std::move(*this);
                r.op_state.release();

                if constexpr (std::is_same_v<std::decay_t<Error>, std::exception_ptr>)
                {
                    std::rethrow_exception(std::forward<Error>(error));
                }

                PIKA_ASSERT_MSG(false,
                    "set_error was called on the receiver of start_detached, terminating. If you "
                    "want to allow errors from the predecessor sender, handle them first with e.g. "
                    "let_error.");
                std::terminate();
            }

            friend void tag_invoke(
                pika::execution::experimental::set_stopped_t, start_detached_receiver&& r) noexcept
            {
                r.op_state.release();
            };

            template <typename... Ts>
            void set_value(Ts&&...) && noexcept
            {
                auto r = std::move(*this);
                r.op_state.release();
            }
        };

    private:
        using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<
            operation_state_holder>;
        PIKA_NO_UNIQUE_ADDRESS allocator_type alloc;

        using operation_state_type =
            pika::execution::experimental::connect_result_t<Sender, start_detached_receiver>;
        std::decay_t<operation_state_type> op_state;

    public:
        template <typename Sender_,
            typename = std::enable_if_t<
                !std::is_same<std::decay_t<Sender_>, operation_state_holder>::value>>
        explicit operation_state_holder(Sender_&& sender, allocator_type const& alloc)
          : alloc(alloc)
          , op_state(pika::execution::experimental::connect(
                std::forward<Sender_>(sender), start_detached_receiver{*this}))
        {
            pika::execution::experimental::start(op_state);
        }

        void release() noexcept
        {
            allocator_type other_alloc(alloc);
            std::allocator_traits<allocator_type>::destroy(other_alloc, this);
            std::allocator_traits<allocator_type>::deallocate(other_alloc, this, 1);
        }
    };
}    // namespace pika::start_detached_detail

namespace pika::execution::experimental {
    inline constexpr struct start_detached_t final
      : pika::functional::detail::tag_fallback<start_detached_t>
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
        friend constexpr PIKA_FORCEINLINE void tag_fallback_invoke(
            start_detached_t, Sender&& sender, Allocator const& allocator = Allocator{})
        {
            using allocator_type = Allocator;
            using operation_state_type =
                start_detached_detail::operation_state_holder<Sender, Allocator>;
            using other_allocator = typename std::allocator_traits<
                allocator_type>::template rebind_alloc<operation_state_type>;
            using allocator_traits = std::allocator_traits<other_allocator>;
            using unique_ptr = std::unique_ptr<operation_state_type,
                pika::detail::allocator_deleter<other_allocator>>;

            other_allocator alloc(allocator);
            unique_ptr p(allocator_traits::allocate(alloc, 1),
                pika::detail::allocator_deleter<other_allocator>{alloc});

            new (p.get()) operation_state_type{std::forward<Sender>(sender), alloc};
            PIKA_UNUSED(p.release());
        }
    } start_detached{};
}    // namespace pika::execution::experimental
