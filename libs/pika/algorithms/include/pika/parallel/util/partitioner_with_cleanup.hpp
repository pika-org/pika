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

#include <pika/execution/executors/execution.hpp>
#include <pika/execution/executors/execution_parameters.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/parallel/util/detail/chunk_size.hpp>
#include <pika/parallel/util/detail/handle_local_exceptions.hpp>
#include <pika/parallel/util/detail/scoped_executor_parameters.hpp>
#include <pika/parallel/util/detail/select_partitioner.hpp>
#include <pika/parallel/util/partitioner.hpp>

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
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        // The static partitioner with cleanup spawns several chunks of
        // iterations for each available core. The number of iterations is
        // determined automatically based on the measured runtime of the
        // iterations.
        template <typename ExPolicy, typename R, typename Result>
        struct static_partitioner_with_cleanup
        {
            using parameters_type = typename ExPolicy::executor_parameters_type;
            using executor_type = typename ExPolicy::executor_type;

            using scoped_executor_parameters =
                detail::scoped_executor_parameters_ref<parameters_type,
                    executor_type>;

            using handle_local_exceptions =
                detail::handle_local_exceptions<ExPolicy>;

            template <typename ExPolicy_, typename FwdIter, typename F1,
                typename F2, typename Cleanup>
            static R call(ExPolicy_&& policy, FwdIter first, std::size_t count,
                F1&& f1, F2&& f2, Cleanup&& cleanup)
            {
                // inform parameter traits
                scoped_executor_parameters scoped_params(
                    policy.parameters(), policy.executor());

                std::vector<pika::future<Result>> workitems;
                std::list<std::exception_ptr> errors;
                try
                {
                    workitems = detail::partition<Result>(
                        PIKA_FORWARD(ExPolicy_, policy), first, count,
                        PIKA_FORWARD(F1, f1));

                    scoped_params.mark_end_of_scheduling();
                }
                catch (...)
                {
                    handle_local_exceptions::call(
                        std::current_exception(), errors);
                }
                return reduce(PIKA_MOVE(workitems), PIKA_MOVE(errors),
                    PIKA_FORWARD(F2, f2), PIKA_FORWARD(Cleanup, cleanup));
            }

        private:
            template <typename F, typename Cleanup>
            static R reduce(std::vector<pika::future<Result>>&& workitems,
                std::list<std::exception_ptr>&& errors, F&& f,
                Cleanup&& cleanup)
            {
                // wait for all tasks to finish
                pika::wait_all_nothrow(workitems);

                // always rethrow if 'errors' is not empty or workitems has
                // exceptional future
                handle_local_exceptions::call_with_cleanup(
                    workitems, errors, PIKA_FORWARD(Cleanup, cleanup));

                try
                {
                    return f(PIKA_MOVE(workitems));
                }
                catch (...)
                {
                    // rethrow either bad_alloc or exception_list
                    handle_local_exceptions::call(std::current_exception());
                    PIKA_ASSERT(false);
                    return f(PIKA_MOVE(workitems));
                }
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename R, typename Result>
        struct task_static_partitioner_with_cleanup
        {
            using parameters_type = typename ExPolicy::executor_parameters_type;
            using executor_type = typename ExPolicy::executor_type;

            using scoped_executor_parameters =
                detail::scoped_executor_parameters<parameters_type,
                    executor_type>;

            using handle_local_exceptions =
                detail::handle_local_exceptions<ExPolicy>;

            template <typename ExPolicy_, typename FwdIter, typename F1,
                typename F2, typename Cleanup>
            static pika::future<R> call(ExPolicy_&& policy, FwdIter first,
                std::size_t count, F1&& f1, F2&& f2, Cleanup&& cleanup)
            {
                // inform parameter traits
                std::shared_ptr<scoped_executor_parameters> scoped_params =
                    std::make_shared<scoped_executor_parameters>(
                        policy.parameters(), policy.executor());

                std::vector<pika::future<Result>> workitems;
                std::list<std::exception_ptr> errors;
                try
                {
                    workitems = detail::partition<Result>(
                        PIKA_FORWARD(ExPolicy_, policy), first, count,
                        PIKA_FORWARD(F1, f1));

                    scoped_params->mark_end_of_scheduling();
                }
                catch (std::bad_alloc const&)
                {
                    return pika::make_exceptional_future<R>(
                        std::current_exception());
                }
                catch (...)
                {
                    handle_local_exceptions::call(
                        std::current_exception(), errors);
                }
                return reduce(PIKA_MOVE(scoped_params), PIKA_MOVE(workitems),
                    PIKA_MOVE(errors), PIKA_FORWARD(F2, f2),
                    PIKA_FORWARD(Cleanup, cleanup));
            }

        private:
            template <typename F, typename Cleanup>
            static pika::future<R> reduce(
                std::shared_ptr<scoped_executor_parameters>&& scoped_params,
                std::vector<pika::future<Result>>&& workitems,
                std::list<std::exception_ptr>&& errors, F&& f,
                Cleanup&& cleanup)
            {
                // wait for all tasks to finish
#if defined(PIKA_COMPUTE_DEVICE_CODE)
                PIKA_UNUSED(scoped_params);
                PIKA_UNUSED(workitems);
                PIKA_UNUSED(errors);
                PIKA_UNUSED(f);
                PIKA_UNUSED(cleanup);
                PIKA_ASSERT(false);
                return pika::future<R>{};
#else
                return pika::dataflow(
                    [errors = PIKA_MOVE(errors),
                        scoped_params = PIKA_MOVE(scoped_params),
                        f = PIKA_FORWARD(F, f),
                        cleanup = PIKA_FORWARD(Cleanup, cleanup)](
                        std::vector<pika::future<Result>>&& r) mutable -> R {
                        PIKA_UNUSED(scoped_params);

                        handle_local_exceptions::call_with_cleanup(
                            r, errors, PIKA_FORWARD(Cleanup, cleanup));
                        return f(PIKA_MOVE(r));
                    },
                    PIKA_MOVE(workitems));
#endif
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // ExPolicy: execution policy
    // R:        overall result type
    // Result:   intermediate result type of first step
    template <typename ExPolicy, typename R = void, typename Result = R>
    struct partitioner_with_cleanup
      : detail::select_partitioner<typename std::decay<ExPolicy>::type,
            detail::static_partitioner_with_cleanup,
            detail::task_static_partitioner_with_cleanup>::template apply<R,
            Result>
    {
    };
}}}    // namespace pika::parallel::util
