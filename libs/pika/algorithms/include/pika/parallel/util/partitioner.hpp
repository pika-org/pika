//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#if !defined(PIKA_COMPUTE_DEVICE_CODE)
#include <pika/async_local/dataflow.hpp>
#endif
#include <pika/async_combinators/wait_all.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/iterator_support/range.hpp>
#include <pika/modules/errors.hpp>
#include <pika/type_support/empty_function.hpp>
#include <pika/type_support/unused.hpp>

#include <pika/execution/executors/execution.hpp>
#include <pika/execution/executors/execution_parameters.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/parallel/util/detail/chunk_size.hpp>
#include <pika/parallel/util/detail/handle_local_exceptions.hpp>
#include <pika/parallel/util/detail/partitioner_iteration.hpp>
#include <pika/parallel/util/detail/scoped_executor_parameters.hpp>
#include <pika/parallel/util/detail/select_partitioner.hpp>

#include <cstddef>
#include <exception>
#include <iterator>
#include <list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace parallel { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Result, typename ExPolicy, typename FwdIter,
            typename F>
        std::vector<pika::future<Result>> partition(
            ExPolicy&& policy, FwdIter first, std::size_t count, F&& f)
        {
            // estimate a chunk size based on number of cores used
            using parameters_type =
                typename std::decay<ExPolicy>::type::executor_parameters_type;
            using has_variable_chunk_size =
                typename execution::extract_has_variable_chunk_size<
                    parameters_type>::type;

            std::vector<pika::future<Result>> inititems;
            auto shape = detail::get_bulk_iteration_shape(
                has_variable_chunk_size{}, PIKA_FORWARD(ExPolicy, policy),
                inititems, f, first, count, 1);

            std::vector<pika::future<Result>> workitems =
                execution::bulk_async_execute(policy.executor(),
                    partitioner_iteration<Result, F>{PIKA_FORWARD(F, f)},
                    PIKA_MOVE(shape));

            if (inititems.empty())
                return workitems;

            // add the newly created workitems to the list
            inititems.insert(inititems.end(),
                std::make_move_iterator(workitems.begin()),
                std::make_move_iterator(workitems.end()));
            return inititems;
        }

        template <typename Result, typename ExPolicy, typename FwdIter,
            typename Stride, typename F>
        std::vector<pika::future<Result>> partition_with_index(ExPolicy&& policy,
            FwdIter first, std::size_t count, Stride stride, F&& f)
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
                inititems, f, first, count, stride);

            std::vector<pika::future<Result>> workitems =
                execution::bulk_async_execute(policy.executor(),
                    partitioner_iteration<Result, F>{PIKA_FORWARD(F, f)},
                    PIKA_MOVE(shape));

            if (inititems.empty())
                return workitems;

            // add the newly created workitems to the list
            inititems.insert(inititems.end(),
                std::make_move_iterator(workitems.begin()),
                std::make_move_iterator(workitems.end()));
            return inititems;
        }

        template <typename Result, typename ExPolicy, typename FwdIter,
            typename Data, typename F>
        // requires is_container<Data>
        std::vector<pika::future<Result>> partition_with_data(ExPolicy&& policy,
            FwdIter first, std::size_t count,
            std::vector<std::size_t> const& chunk_sizes, Data&& data, F&& f)
        {
            PIKA_ASSERT(pika::util::size(data) >= pika::util::size(chunk_sizes));

            typedef typename std::decay<Data>::type data_type;

            typename data_type::const_iterator data_it = pika::util::begin(data);
            typename std::vector<std::size_t>::const_iterator chunk_size_it =
                pika::util::begin(chunk_sizes);

            typedef typename pika::tuple<typename data_type::value_type, FwdIter,
                std::size_t>
                tuple_type;

            // schedule every chunk on a separate thread
            std::vector<tuple_type> shape;
            shape.reserve(chunk_sizes.size());

            while (count != 0)
            {
                std::size_t chunk = (std::min)(count, *chunk_size_it);
                PIKA_ASSERT(chunk != 0);

                shape.push_back(pika::make_tuple(*data_it, first, chunk));

                count -= chunk;
                std::advance(first, chunk);

                ++data_it;
                ++chunk_size_it;
            }
            PIKA_ASSERT(chunk_size_it == chunk_sizes.end());

            return execution::bulk_async_execute(policy.executor(),
                partitioner_iteration<Result, F>{PIKA_FORWARD(F, f)},
                PIKA_MOVE(shape));
        }

        ///////////////////////////////////////////////////////////////////////
        // The static partitioner simply spawns one chunk of iterations for
        // each available core.
        template <typename ExPolicy, typename R, typename Result>
        struct static_partitioner
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
            static R call(ExPolicy_&& policy, FwdIter first, std::size_t count,
                F1&& f1, F2&& f2)
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
                return reduce(
                    PIKA_MOVE(workitems), PIKA_MOVE(errors), PIKA_FORWARD(F2, f2));
            }

            template <typename ExPolicy_, typename FwdIter, typename Stride,
                typename F1, typename F2>
            static R call_with_index(ExPolicy_&& policy, FwdIter first,
                std::size_t count, Stride stride, F1&& f1, F2&& f2)
            {
                // inform parameter traits
                scoped_executor_parameters scoped_params(
                    policy.parameters(), policy.executor());

                std::vector<pika::future<Result>> workitems;
                std::list<std::exception_ptr> errors;
                try
                {
                    workitems = detail::partition_with_index<Result>(
                        PIKA_FORWARD(ExPolicy_, policy), first, count, stride,
                        PIKA_FORWARD(F1, f1));

                    scoped_params.mark_end_of_scheduling();
                }
                catch (...)
                {
                    handle_local_exceptions::call(
                        std::current_exception(), errors);
                }
                return reduce(
                    PIKA_MOVE(workitems), PIKA_MOVE(errors), PIKA_FORWARD(F2, f2));
            }

            template <typename ExPolicy_, typename FwdIter, typename F1,
                typename F2, typename Data>
            // requires is_container<Data>
            static R call_with_data(ExPolicy_&& policy, FwdIter first,
                std::size_t count, F1&& f1, F2&& f2,
                std::vector<std::size_t> const& chunk_sizes, Data&& data)
            {
                // inform parameter traits
                scoped_executor_parameters scoped_params(
                    policy.parameters(), policy.executor());

                std::vector<pika::future<Result>> workitems;
                std::list<std::exception_ptr> errors;
                try
                {
                    workitems = detail::partition_with_data<Result>(
                        PIKA_FORWARD(ExPolicy_, policy), first, count,
                        chunk_sizes, PIKA_FORWARD(Data, data),
                        PIKA_FORWARD(F1, f1));

                    scoped_params.mark_end_of_scheduling();
                }
                catch (...)
                {
                    handle_local_exceptions::call(
                        std::current_exception(), errors);
                }
                return reduce(
                    PIKA_MOVE(workitems), PIKA_MOVE(errors), PIKA_FORWARD(F2, f2));
            }

        private:
            template <typename F>
            static R reduce(std::vector<pika::future<Result>>&& workitems,
                std::list<std::exception_ptr>&& errors, F&& f)
            {
                // wait for all tasks to finish
                pika::wait_all_nothrow(workitems);

                // always rethrow if 'errors' is not empty or workitems has
                // exceptional future
                handle_local_exceptions::call(workitems, errors);

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

            static R reduce(std::vector<pika::future<Result>>&& workitems,
                std::list<std::exception_ptr>&& errors,
                pika::util::empty_function)
            {
                // wait for all tasks to finish
                pika::wait_all_nothrow(workitems);

                // always rethrow if 'errors' is not empty or workitems has
                // exceptional future
                handle_local_exceptions::call(workitems, errors);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename R, typename Result>
        struct task_static_partitioner
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
            static pika::future<R> call(ExPolicy_&& policy, FwdIter first,
                std::size_t count, F1&& f1, F2&& f2)
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
                    PIKA_MOVE(errors), PIKA_FORWARD(F2, f2));
            }

            template <typename ExPolicy_, typename FwdIter, typename Stride,
                typename F1, typename F2>
            static pika::future<R> call_with_index(ExPolicy_&& policy,
                FwdIter first, std::size_t count, Stride stride, F1&& f1,
                F2&& f2)
            {
                // inform parameter traits
                std::shared_ptr<scoped_executor_parameters> scoped_params =
                    std::make_shared<scoped_executor_parameters>(
                        policy.parameters(), policy.executor());

                std::vector<pika::future<Result>> workitems;
                std::list<std::exception_ptr> errors;
                try
                {
                    workitems = detail::partition_with_index<Result>(
                        PIKA_FORWARD(ExPolicy_, policy), first, count, stride,
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
                    PIKA_MOVE(errors), PIKA_FORWARD(F2, f2));
            }

            template <typename ExPolicy_, typename FwdIter, typename F1,
                typename F2, typename Data>
            // requires is_container<Data>
            static pika::future<R> call_with_data(ExPolicy_&& policy,
                FwdIter first, std::size_t count, F1&& f1, F2&& f2,
                std::vector<std::size_t> const& chunk_sizes, Data&& data)
            {
                // inform parameter traits
                std::shared_ptr<scoped_executor_parameters> scoped_params =
                    std::make_shared<scoped_executor_parameters>(
                        policy.parameters(), policy.executor());

                std::vector<pika::future<Result>> workitems;
                std::list<std::exception_ptr> errors;
                try
                {
                    workitems = detail::partition_with_data<Result>(
                        PIKA_FORWARD(ExPolicy_, policy), first, count,
                        chunk_sizes, PIKA_FORWARD(Data, data),
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
                    PIKA_MOVE(errors), PIKA_FORWARD(F2, f2));
            }

        private:
            template <typename F>
            static pika::future<R> reduce(
                std::shared_ptr<scoped_executor_parameters>&& scoped_params,
                std::vector<pika::future<Result>>&& workitems,
                std::list<std::exception_ptr>&& errors, F&& f)
            {
#if defined(PIKA_COMPUTE_DEVICE_CODE)
                PIKA_UNUSED(scoped_params);
                PIKA_UNUSED(workitems);
                PIKA_UNUSED(errors);
                PIKA_UNUSED(f);
                PIKA_ASSERT(false);
                return pika::future<R>();
#else
                // wait for all tasks to finish
                return pika::dataflow(
                    [errors = PIKA_MOVE(errors),
                        scoped_params = PIKA_MOVE(scoped_params),
                        f = PIKA_FORWARD(F, f)](
                        std::vector<pika::future<Result>>&& r) mutable -> R {
                        PIKA_UNUSED(scoped_params);

                        handle_local_exceptions::call(r, errors);
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
    struct partitioner
      : detail::select_partitioner<typename std::decay<ExPolicy>::type,
            detail::static_partitioner,
            detail::task_static_partitioner>::template apply<R, Result>
    {
    };
}}}    // namespace pika::parallel::util
