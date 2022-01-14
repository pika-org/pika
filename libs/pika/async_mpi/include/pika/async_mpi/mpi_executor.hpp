//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/async_mpi/mpi_future.hpp>
#include <pika/execution/executors/static_chunk_size.hpp>
#include <pika/execution_base/execution.hpp>
#include <pika/execution_base/traits/is_executor.hpp>
#include <pika/modules/mpi_base.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace pika { namespace mpi { namespace experimental {

    struct executor
    {
        // Associate the parallel_execution_tag executor tag type as a default
        // with this executor.
        using execution_category = pika::execution::parallel_execution_tag;

        // default params type as we don't do anything special
        using executor_parameters_type = pika::execution::static_chunk_size;

        constexpr executor(MPI_Comm communicator = MPI_COMM_WORLD)
          : communicator_(communicator)
        {
        }

        /// \cond NOINTERNAL
        constexpr bool operator==(executor const& rhs) const noexcept
        {
            return communicator_ == rhs.communicator_;
        }

        constexpr bool operator!=(executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        constexpr executor const& context() const noexcept
        {
            return *this;
        }
        /// \endcond

        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        decltype(auto) async_execute(F&& f, Ts&&... ts) const
        {
            return pika::mpi::experimental::detail::async(
                PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)..., communicator_);
        }

        std::size_t in_flight_estimate() const
        {
            return detail::get_mpi_info().requests_vector_size_ +
                detail::get_mpi_info().requests_queue_size_;
        }

    private:
        MPI_Comm communicator_;
    };
}}}    // namespace pika::mpi::experimental

namespace pika { namespace parallel { namespace execution {

    /// \cond NOINTERNAL
    template <>
    struct is_two_way_executor<pika::mpi::experimental::executor>
      : std::true_type
    {
    };
    /// \endcond
}}}    // namespace pika::parallel::execution
