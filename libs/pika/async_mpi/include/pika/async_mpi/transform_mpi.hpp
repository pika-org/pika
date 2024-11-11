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
#include <pika/execution/algorithms/continues_on.hpp>
#include <pika/execution/algorithms/detail/helpers.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution/algorithms/just.hpp>
#include <pika/execution/algorithms/let_value.hpp>
#include <pika/execution/algorithms/unpack.hpp>
#include <pika/execution_base/any_sender.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
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
            using pika::execution::experimental::continues_on;
            using pika::execution::experimental::just;
            using pika::execution::experimental::let_value;
            using pika::execution::experimental::unique_any_sender;
            using pika::execution::experimental::unpack;

            // get mpi completion mode settings
            auto mode = get_completion_mode();
            bool completions_inline = use_inline_completion(mode);
            bool requests_inline = use_inline_request(mode);

            execution::thread_priority p = use_priority_boost(mode) ?
                execution::thread_priority::boost :
                execution::thread_priority::normal;

            auto completion_snd = [=](MPI_Request request) -> unique_any_sender<> {
                if (!completions_inline)    // not inline : a transfer is required
                {
                    if (request == MPI_REQUEST_NULL)
                    {
                        return ex::schedule(default_pool_scheduler(p));
                    }
                    return just(request) | trigger_mpi(mode) |
                        ex::continues_on(default_pool_scheduler(p));
                }
                if (request == MPI_REQUEST_NULL) { return just(); }
                return just(request) | trigger_mpi(mode);
            };

            if (requests_inline)
            {
                return std::forward<Sender>(sender) |
                    let_value([=, f = std::forward<F>(f)](auto&&... args) mutable {
                        std::tuple<decltype(args)&...> ts{args...};
                        auto snd0 = just(ts) | ex::unpack();
                        return dispatch_mpi_sender<decltype(snd0), F>{
                                   std::move(snd0), std::move(f)} |
                            let_value(completion_snd);
                    });
            }
            else
            {
                return std::forward<Sender>(sender) | continues_on(mpi_pool_scheduler(p)) |
                    let_value([=, f = std::forward<F>(f)](auto&... args) mutable {
                        std::tuple<decltype(args)&...> ts{args...};
                        auto snd0 = just(ts) | ex::unpack();
                        return dispatch_mpi_sender<decltype(snd0), F>{
                                   std::move(snd0), std::move(f)} |
                            let_value(completion_snd);
                    });
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
