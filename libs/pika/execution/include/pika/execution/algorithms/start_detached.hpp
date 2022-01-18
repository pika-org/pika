//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/allocator_support/allocator_deleter.hpp>
#include <pika/allocator_support/internal_allocator.hpp>
#include <pika/allocator_support/traits/is_allocator.hpp>
#include <pika/assert.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution_base/operation_state.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/invoke_result.hpp>
#include <pika/modules/memory.hpp>
#include <pika/thread_support/atomic_count.hpp>
#include <pika/type_support/unused.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

namespace pika { namespace execution { namespace experimental {
    namespace detail {
        template <typename Sender, typename Allocator>
        struct operation_state_holder
        {
            struct start_detached_receiver
            {
                pika::intrusive_ptr<operation_state_holder> op_state;

                template <typename Error>
                PIKA_NORETURN friend void tag_invoke(
                    set_error_t, start_detached_receiver&&, Error&&) noexcept
                {
                    PIKA_ASSERT_MSG(false,
                        "set_error was called on the receiver of "
                        "start_detached, "
                        "terminating. If you want to allow errors from the "
                        "predecessor sender, handle them first with e.g. "
                        "let_error.");
                    std::terminate();
                }

                friend void tag_invoke(
                    set_done_t, start_detached_receiver&& r) noexcept
                {
                    r.op_state.reset();
                };

                template <typename... Ts>
                friend void tag_invoke(
                    set_value_t, start_detached_receiver&& r, Ts&&...) noexcept
                {
                    r.op_state.reset();
                }
            };

        private:
            using allocator_type = typename std::allocator_traits<
                Allocator>::template rebind_alloc<operation_state_holder>;
            PIKA_NO_UNIQUE_ADDRESS allocator_type alloc;
            pika::util::atomic_count count{0};

            using operation_state_type =
                connect_result_t<Sender, start_detached_receiver>;
            std::decay_t<operation_state_type> op_state;

        public:
            template <typename Sender_,
                typename = std::enable_if_t<!std::is_same<std::decay_t<Sender_>,
                    operation_state_holder>::value>>
            explicit operation_state_holder(
                Sender_&& sender, allocator_type const& alloc)
              : alloc(alloc)
              , op_state(connect(PIKA_FORWARD(Sender_, sender),
                    start_detached_receiver{this}))
            {
                start(op_state);
            }

        private:
            friend void intrusive_ptr_add_ref(operation_state_holder* p)
            {
                ++p->count;
            }

            friend void intrusive_ptr_release(operation_state_holder* p)
            {
                if (--p->count == 0)
                {
                    allocator_type other_alloc(p->alloc);
                    std::allocator_traits<allocator_type>::destroy(
                        other_alloc, p);
                    std::allocator_traits<allocator_type>::deallocate(
                        other_alloc, p, 1);
                }
            }
        };
    }    // namespace detail

    inline constexpr struct start_detached_t final
      : pika::functional::detail::tag_fallback<start_detached_t>
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
        friend constexpr PIKA_FORCEINLINE void tag_fallback_invoke(
            start_detached_t, Sender&& sender,
            Allocator const& allocator = Allocator{})
        {
            using allocator_type = Allocator;
            using operation_state_type =
                detail::operation_state_holder<Sender, Allocator>;
            using other_allocator = typename std::allocator_traits<
                allocator_type>::template rebind_alloc<operation_state_type>;
            using allocator_traits = std::allocator_traits<other_allocator>;
            using unique_ptr = std::unique_ptr<operation_state_type,
                util::allocator_deleter<other_allocator>>;

            other_allocator alloc(allocator);
            unique_ptr p(allocator_traits::allocate(alloc, 1),
                pika::util::allocator_deleter<other_allocator>{alloc});

            new (p.get())
                operation_state_type{PIKA_FORWARD(Sender, sender), alloc};
            PIKA_UNUSED(p.release());
        }

        // clang-format off
        template <typename Allocator = pika::util::internal_allocator<>,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            start_detached_t, Allocator const& allocator = Allocator{})
        {
            return detail::partial_algorithm<start_detached_t, Allocator>{
                allocator};
        }
    } start_detached{};
}}}    // namespace pika::execution::experimental
