//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/transform_xxx.hpp

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/async_mpi/dispatch_mpi.hpp>
#include <pika/async_mpi/mpi_polling.hpp>
#include <pika/async_mpi/trigger_mpi.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/debugging/print.hpp>
#include <pika/execution/algorithms/detail/helpers.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution/algorithms/let_value.hpp>
#include <pika/execution/algorithms/transfer.hpp>
#include <pika/execution/algorithms/transfer_just.hpp>
#include <pika/execution_base/any_sender.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/executors/thread_pool_scheduler_queue_bypass.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/mpi_base/mpi.hpp>

#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pika::mpi::experimental {

    inline constexpr struct transform_mpi_t final
      : pika::functional::detail::tag_fallback<transform_mpi_t>
    {
    private:
        template <typename Sender, typename F,
            PIKA_CONCEPT_REQUIRES_(
                pika::execution::experimental::is_sender_v<std::decay_t<Sender>>)>
        friend PIKA_FORCEINLINE pika::execution::experimental::unique_any_sender<>
        tag_fallback_invoke(transform_mpi_t, Sender&& sender, F&& f)
        {
            using namespace pika::mpi::experimental::detail;
            PIKA_DETAIL_DP(mpi_tran<5>, debug(str<>("transform_mpi_t"), "tag_fallback_invoke"));

            using execution::thread_priority;
            using pika::execution::experimental::just;
            using pika::execution::experimental::let_value;
            using pika::execution::experimental::thread_pool_scheduler_queue_bypass;
            using pika::execution::experimental::transfer;
            using pika::execution::experimental::transfer_just;
            using pika::execution::experimental::unique_any_sender;

            // get mpi completion mode settings
            auto mode = get_completion_mode();
            bool completions_inline = use_inline_completion(mode);
            bool requests_inline = use_inline_request(mode);

#ifdef PIKA_DEBUG
            // ----------------------------------------------------------
            // the pool should exist if the completion mode needs it
            int cwsize = detail::comm_world_size();
            bool need_pool = (cwsize > 1 && use_pool(mode));
            if (pool_exists() != need_pool)
            {
                PIKA_DETAIL_DP(mpi_tran<0>,
                    error(str<>("transform_mpi"), "mode", mode, "pool_exists()", pool_exists(),
                        "need_pool", need_pool));
            }
            PIKA_ASSERT(pool_exists() == need_pool);
#endif

            execution::thread_priority p = use_priority_boost(mode) ?
                execution::thread_priority::boost :
                execution::thread_priority::normal;

            auto dgb = []() {
                PIKA_DETAIL_DP(mpi_tran<1>, debug(str<>("transform_mpi"), "complete"));
            };
            auto completion_snd = [=](MPI_Request request) -> unique_any_sender<> {
                if (!completions_inline)    // not inline : a transfer is required
                {
                    if (request == MPI_REQUEST_NULL)
                    {
                        return transfer_just(default_pool_scheduler(p));
                    }
                    return just(request) | trigger_mpi(mode) | transfer(default_pool_scheduler(p));
                }
                // inline, can we skip any steps?
                if (request == MPI_REQUEST_NULL) { return just(); }
                if (inline_ready(mode))    // the completion mode does not require a transfer
                {
                    return just(request) | trigger_mpi(mode);    //
                }
                return just(request) | trigger_mpi(mode) |
                    transfer(ex::with_annotation(
                        ex::thread_pool_scheduler_queue_bypass{}, "transform_mpi"));
            };

            if (requests_inline)
            {
                return dispatch_mpi(PIKA_MOVE(sender), PIKA_FORWARD(F, f)) |
#ifdef PIKA_DEBUG
                    let_value(completion_snd) | ex::then(dgb);
#else
                    let_value(completion_snd);
#endif
            }
            else
            {
                auto snd0 = PIKA_FORWARD(Sender, sender) | transfer(mpi_pool_scheduler(p));
                return dispatch_mpi(PIKA_MOVE(snd0), PIKA_FORWARD(F, f)) |
#ifdef PIKA_DEBUG
                    let_value(completion_snd) | ex::then(dgb);
#else
                    let_value(completion_snd);
#endif
            }
        }

        //
        // tag invoke overload for mpi_transform
        //
        template <typename F>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(transform_mpi_t, F&& f)
        {
            return pika::execution::experimental::detail::partial_algorithm<transform_mpi_t, F>{
                PIKA_FORWARD(F, f)};
        }

    } transform_mpi{};
}    // namespace pika::mpi::experimental
