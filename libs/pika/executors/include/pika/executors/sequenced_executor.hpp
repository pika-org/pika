//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/sequenced_executor.hpp

#pragma once

#include <pika/local/config.hpp>
#include <pika/errors/exception_list.hpp>
#include <pika/execution/detail/async_launch_policy_dispatch.hpp>
#include <pika/execution/detail/sync_launch_policy_dispatch.hpp>
#include <pika/execution/executors/execution.hpp>
#include <pika/execution_base/traits/is_executor.hpp>
#include <pika/functional/deferred_call.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/futures/future.hpp>
#include <pika/pack_traversal/unwrap.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    /// A \a sequential_executor creates groups of sequential execution agents
    /// which execute in the calling thread. The sequential order is given by
    /// the lexicographical order of indices in the index space.
    ///
    struct sequenced_executor
    {
        /// \cond NOINTERNAL
        bool operator==(sequenced_executor const& /*rhs*/) const noexcept
        {
            return true;
        }

        bool operator!=(sequenced_executor const& /*rhs*/) const noexcept
        {
            return false;
        }

        sequenced_executor const& context() const noexcept
        {
            return *this;
        }
        /// \endcond

        /// \cond NOINTERNAL
        typedef sequenced_execution_tag execution_category;

        // OneWayExecutor interface
        template <typename F, typename... Ts>
        static
            typename pika::util::detail::invoke_deferred_result<F, Ts...>::type
            sync_execute(F&& f, Ts&&... ts)
        {
            return pika::detail::sync_launch_policy_dispatch<
                launch::sync_policy>::call(launch::sync, PIKA_FORWARD(F, f),
                PIKA_FORWARD(Ts, ts)...);
        }

        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        static pika::future<
            typename pika::util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F&& f, Ts&&... ts)
        {
            return pika::detail::async_launch_policy_dispatch<
                launch::deferred_policy>::call(launch::deferred,
                PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
        }

        // NonBlockingOneWayExecutor (adapted) interface
        template <typename F, typename... Ts>
        static void post(F&& f, Ts&&... ts)
        {
            sync_execute(PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
        }

        // BulkTwoWayExecutor interface
        template <typename F, typename S, typename... Ts>
        static std::vector<pika::future<typename parallel::execution::detail::
                bulk_function_result<F, S, Ts...>::type>>
        bulk_async_execute(F&& f, S const& shape, Ts&&... ts)
        {
            typedef
                typename parallel::execution::detail::bulk_function_result<F, S,
                    Ts...>::type result_type;
            std::vector<pika::future<result_type>> results;

            try
            {
                for (auto const& elem : shape)
                {
                    results.push_back(async_execute(f, elem, ts...));
                }
            }
            catch (std::bad_alloc const& ba)
            {
                throw ba;
            }
            catch (...)
            {
                throw exception_list(std::current_exception());
            }

            return results;
        }

        template <typename F, typename S, typename... Ts>
        static typename parallel::execution::detail::bulk_execute_result<F, S,
            Ts...>::type
        bulk_sync_execute(F&& f, S const& shape, Ts&&... ts)
        {
            return pika::unwrap(bulk_async_execute(
                PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Ts, ts)...));
        }

        template <typename Parameters>
        std::size_t processing_units_count(Parameters&&) const
        {
            return 1;
        }

    private:
        friend class pika::serialization::access;

        template <typename Archive>
        void serialize(Archive& /* ar */, const unsigned int /* version */)
        {
        }
        /// \endcond
    };
}}    // namespace pika::execution

namespace pika { namespace parallel { namespace execution {
    using sequenced_executor PIKA_DEPRECATED_V(0, 1,
        "pika::parallel::execution::sequenced_executor is deprecated. Use "
        "pika::execution::sequenced_executor instead.") =
        pika::execution::sequenced_executor;
}}}    // namespace pika::parallel::execution

namespace pika { namespace parallel { namespace execution {
    /// \cond NOINTERNAL
    template <>
    struct is_one_way_executor<pika::execution::sequenced_executor>
      : std::true_type
    {
    };

    template <>
    struct is_bulk_one_way_executor<pika::execution::sequenced_executor>
      : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<pika::execution::sequenced_executor>
      : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<pika::execution::sequenced_executor>
      : std::true_type
    {
    };
    /// \endcond
}}}    // namespace pika::parallel::execution
