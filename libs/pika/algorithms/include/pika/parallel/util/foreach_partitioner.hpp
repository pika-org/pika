//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/async_combinators/wait_all.hpp>
#include <pika/modules/errors.hpp>
#if !defined(PIKA_COMPUTE_DEVICE_CODE)
#include <pika/async_local/dataflow.hpp>
#endif
#include <pika/type_support/unused.hpp>

#include <pika/execution/algorithms/detail/predicates.hpp>
#include <pika/execution/executors/execution.hpp>
#include <pika/execution/executors/execution_parameters.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/parallel/util/detail/chunk_size.hpp>
#include <pika/parallel/util/detail/handle_local_exceptions.hpp>
#include <pika/parallel/util/detail/partitioner_iteration.hpp>
#include <pika/parallel/util/detail/scoped_executor_parameters.hpp>
#include <pika/parallel/util/detail/select_partitioner.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace parallel { namespace util {
    namespace detail {
        template <typename Result, typename ExPolicy, typename FwdIter,
            typename F>
        std::pair<std::vector<pika::future<Result>>,
            std::vector<pika::future<Result>>>
        foreach_partition(
            ExPolicy&& policy, FwdIter first, std::size_t count, F&& f)
        {
            // estimate a chunk size based on number of cores used
            using parameters_type =
                typename std::decay<ExPolicy>::type::executor_parameters_type;
            using has_variable_chunk_size =
                typename execution::extract_has_variable_chunk_size<
                    parameters_type>::type;

            std::vector<pika::future<Result>> inititems;
            auto shape = detail::get_bulk_iteration_shape_idx(
                has_variable_chunk_size{}, PIKA_FORWARD(ExPolicy, policy),
                inititems, f, first, count, 1);

            std::vector<pika::future<Result>> workitems =
                execution::bulk_async_execute(policy.executor(),
                    partitioner_iteration<Result, F>{PIKA_FORWARD(F, f)},
                    PIKA_MOVE(shape));
            return std::make_pair(PIKA_MOVE(inititems), PIKA_MOVE(workitems));
        }

        ///////////////////////////////////////////////////////////////////////
        // The static partitioner simply spawns one chunk of iterations for
        // each available core.
        template <typename ExPolicy, typename Result>
        struct foreach_static_partitioner
        {
            using parameters_type = typename ExPolicy::executor_parameters_type;
            using executor_type = typename ExPolicy::executor_type;

            using scoped_executor_parameters =
                detail::scoped_executor_parameters_ref<parameters_type,
                    executor_type>;

            using handle_local_exceptions =
                detail::handle_local_exceptions<ExPolicy>;

            template <typename ExPolicy_, typename FwdIter, typename F1,
                typename F2>
            static FwdIter call(ExPolicy_&& policy, FwdIter first,
                std::size_t count, F1&& f1, F2&& f2)
            {
                // inform parameter traits
                scoped_executor_parameters scoped_params(
                    policy.parameters(), policy.executor());

                FwdIter last = parallel::v1::detail::next(first, count);

                std::vector<pika::future<Result>> inititems, workitems;
                std::list<std::exception_ptr> errors;
                try
                {
                    std::tie(inititems, workitems) =
                        detail::foreach_partition<Result>(
                            PIKA_FORWARD(ExPolicy_, policy), first, count,
                            PIKA_FORWARD(F1, f1));

                    scoped_params.mark_end_of_scheduling();
                }
                catch (...)
                {
                    handle_local_exceptions::call(
                        std::current_exception(), errors);
                }
                return reduce(PIKA_MOVE(inititems), PIKA_MOVE(workitems),
                    PIKA_MOVE(errors), PIKA_FORWARD(F2, f2), PIKA_MOVE(last));
            }

        private:
            template <typename F, typename FwdIter>
            static FwdIter reduce(std::vector<pika::future<Result>>&& inititems,
                std::vector<pika::future<Result>>&& workitems,
                std::list<std::exception_ptr>&& errors, F&& f, FwdIter last)
            {
                // wait for all tasks to finish
                pika::wait_all_nothrow(workitems);

                // always rethrow if 'errors' is not empty or workitems has
                // exceptional future
                handle_local_exceptions::call(inititems, errors);
                handle_local_exceptions::call(workitems, errors);

                try
                {
                    return f(PIKA_MOVE(last));
                }
                catch (...)
                {
                    // rethrow either bad_alloc or exception_list
                    handle_local_exceptions::call(std::current_exception());
                }

                PIKA_ASSERT(false);
                return last;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Result>
        struct foreach_task_static_partitioner
        {
            using parameters_type = typename ExPolicy::executor_parameters_type;
            using executor_type = typename ExPolicy::executor_type;

            using scoped_executor_parameters =
                detail::scoped_executor_parameters<parameters_type,
                    executor_type>;

            using handle_local_exceptions =
                detail::handle_local_exceptions<ExPolicy>;

            template <typename ExPolicy_, typename FwdIter, typename F1,
                typename F2>
            static pika::future<FwdIter> call(ExPolicy_&& policy, FwdIter first,
                std::size_t count, F1&& f1, F2&& f2)
            {
                // inform parameter traits
                std::shared_ptr<scoped_executor_parameters> scoped_params =
                    std::make_shared<scoped_executor_parameters>(
                        policy.parameters(), policy.executor());

                FwdIter last = parallel::v1::detail::next(first, count);

                std::vector<pika::future<Result>> inititems, workitems;
                std::list<std::exception_ptr> errors;
                try
                {
                    std::tie(inititems, workitems) =
                        detail::foreach_partition<Result>(
                            PIKA_FORWARD(ExPolicy_, policy), first, count,
                            PIKA_FORWARD(F1, f1));

                    scoped_params->mark_end_of_scheduling();
                }
                catch (std::bad_alloc const&)
                {
                    return pika::make_exceptional_future<FwdIter>(
                        std::current_exception());
                }
                catch (...)
                {
                    handle_local_exceptions::call(
                        std::current_exception(), errors);
                }
                return reduce(PIKA_MOVE(scoped_params), PIKA_MOVE(inititems),
                    PIKA_MOVE(workitems), PIKA_MOVE(errors), PIKA_FORWARD(F2, f2),
                    PIKA_MOVE(last));
            }

        private:
            template <typename F, typename FwdIter>
            static pika::future<FwdIter> reduce(
                std::shared_ptr<scoped_executor_parameters>&& scoped_params,
                std::vector<pika::future<Result>>&& inititems,
                std::vector<pika::future<Result>>&& workitems,
                std::list<std::exception_ptr>&& errors, F&& f, FwdIter last)
            {
#if defined(PIKA_COMPUTE_DEVICE_CODE)
                PIKA_UNUSED(scoped_params);
                PIKA_UNUSED(inititems);
                PIKA_UNUSED(workitems);
                PIKA_UNUSED(errors);
                PIKA_UNUSED(f);
                PIKA_UNUSED(last);
                PIKA_ASSERT(false);
                return pika::future<FwdIter>();
#else
                // wait for all tasks to finish
                // Note: the lambda takes the vectors by value (dataflow
                //       moves those into the lambda) to ensure that they
                //       will be destroyed before the lambda exists.
                //       Otherwise the vectors stay alive in the dataflow's
                //       shared state and may reference data that has gone
                //       out of scope.
                return pika::dataflow(
                    [last, errors = PIKA_MOVE(errors),
                        scoped_params = PIKA_MOVE(scoped_params),
                        f = PIKA_FORWARD(F, f)](
                        std::vector<pika::future<Result>> r1,
                        std::vector<pika::future<Result>> r2) mutable
                    -> FwdIter {
                        PIKA_UNUSED(scoped_params);

                        handle_local_exceptions::call(r1, errors);
                        handle_local_exceptions::call(r2, errors);
                        return f(PIKA_MOVE(last));
                    },
                    PIKA_MOVE(inititems), PIKA_MOVE(workitems));
#endif
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // ExPolicy: execution policy
    // Result:   intermediate result type of first step (default: void)
    template <typename ExPolicy, typename Result = void>
    struct foreach_partitioner
      : detail::select_partitioner<typename std::decay<ExPolicy>::type,
            detail::foreach_static_partitioner,
            detail::foreach_task_static_partitioner>::template apply<Result>
    {
    };
}}}    // namespace pika::parallel::util
