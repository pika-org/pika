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
#include <pika/execution_base/any_sender.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/functional/invoke_fused.hpp>
#include <pika/mpi_base/mpi.hpp>

#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pika::mpi::experimental {

    namespace pud = pika::util::detail;
    namespace exp = execution::experimental;

    inline constexpr struct transform_mpi_t final
      : pika::functional::detail::tag_fallback<transform_mpi_t>
    {
    private:
        template <typename Sender, typename F,
            PIKA_CONCEPT_REQUIRES_(exp::is_sender_v<std::decay_t<Sender>>)>
        friend constexpr PIKA_FORCEINLINE exp::unique_any_sender<>
        tag_fallback_invoke(transform_mpi_t, Sender&& sender, F&& f,
            /*progress_mode p, */ stream_type s = stream_type::automatic)
        {
            using namespace pika::mpi::experimental::detail;
            using namespace pika::debug::detail;
            PIKA_DETAIL_DP(mpi_tran<5>,
                debug(str<>("transform_mpi_t"), "tag_fallback_invoke", "stream",
                    detail::stream_name(s)));

            using execution::thread_priority;
            using exp::make_unique_any_sender;
            using exp::then;
            using exp::transfer;

            // get the mpi completion mode
            auto mode = get_completion_mode();

            bool inline_com = use_inline_com(mode);
            bool inline_req = use_inline_req(mode);

#ifdef PIKA_DEBUG
            // ----------------------------------------------------------
            // the pool should exist if the completion mode needs it
            int cwsize = detail::comm_world_size();
            bool need_pool = (cwsize > 1 && use_pool(mode));

            if (pool_exists() != need_pool)
            {
                std::cerr << "mode " << mode << " pool_exists() " << pool_exists() << " need_pool "
                          << need_pool << std::endl;
            }
            PIKA_ASSERT(pool_exists() == need_pool);
#endif

            using execution::thread_priority;
            thread_priority p = use_HP_com(mode) ? thread_priority::high : thread_priority::normal;
            if (inline_req)
            {
                return dispatch_mpi_sender<Sender, F>{PIKA_MOVE(sender), PIKA_FORWARD(F, f), s} |
                    pika::execution::experimental::let_value(
                        [=](MPI_Request request) -> exp::unique_any_sender<> {
                            if (inline_com)
                            {
                                if (request == MPI_REQUEST_NULL)
                                    return exp::just();
                                else
                                    return exp::just(request) | trigger_mpi(mode);
                            }
                            else
                            {
                                if (request == MPI_REQUEST_NULL)
                                    return exp::just() | transfer(default_pool_scheduler(p));
                                else
                                    return exp::just(request) |
                                        transfer(default_pool_scheduler(p)) | trigger_mpi(mode);
                            }
                        });
            }
            else
            {
                auto snd0 = PIKA_FORWARD(Sender, sender) | transfer(mpi_pool_scheduler());
                return dispatch_mpi_sender<decltype(snd0), F>{
                           PIKA_MOVE(snd0), PIKA_FORWARD(F, f), s} |
                    exp::let_value([=](MPI_Request request) -> exp::unique_any_sender<> {
                        if (inline_com)
                        {
                            if (request == MPI_REQUEST_NULL)
                                return exp::just();
                            else
                                return exp::just(request) | trigger_mpi(mode);
                        }
                        else
                        {
                            if (request == MPI_REQUEST_NULL)
                                return exp::just() | transfer(default_pool_scheduler(p));
                            else
                                return exp::just(request) | transfer(default_pool_scheduler(p)) |
                                    trigger_mpi(mode);
                        }
                    });
            }
        }

        //
        // tag invoke overload for mpi_transform
        //
        template <typename F>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            transform_mpi_t, F&& f, /*progress_mode p, */ stream_type s = stream_type::automatic)
        {
            return exp::detail::partial_algorithm<transform_mpi_t, F,
                /*progress_mode, */ stream_type>{PIKA_FORWARD(F, f), /*p, */ s};
        }

    } transform_mpi{};
}    // namespace pika::mpi::experimental
