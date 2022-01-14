//  Copyright (c) 2007-2017 Hartmut Kaiser
//                2017-2018 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/executors/execution_policy_fwd.hpp>
#include <pika/functional/function.hpp>
#include <pika/futures/future.hpp>
#include <pika/modules/errors.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace pika { namespace parallel { inline namespace v1 {
    namespace detail {
        /// \cond NOINTERNAL
        template <typename ExPolicy, typename Result = void>
        struct handle_exception_impl
        {
            using type = Result;

            PIKA_NORETURN static Result call()
            {
                try
                {
                    throw;    //-V667
                }
                catch (std::bad_alloc const&)
                {
                    throw;
                }
                catch (pika::exception_list const&)
                {
                    throw;
                }
                catch (...)
                {
                    throw pika::exception_list(std::current_exception());
                }
            }

            static Result call(pika::future<Result> f)
            {
                PIKA_ASSERT(f.has_exception());

                return f.get();
            }

            PIKA_NORETURN static Result call(std::exception_ptr const& e)
            {
                try
                {
                    std::rethrow_exception(e);
                }
                catch (std::bad_alloc const&)
                {
                    throw;
                }
                catch (pika::exception_list const&)
                {
                    throw;
                }
                catch (...)
                {
                    throw pika::exception_list(std::current_exception());
                }
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        struct handle_exception_task_impl
        {
            using type = future<Result>;

            static future<Result> call()
            {
                try
                {
                    throw;    //-V667
                }
                catch (std::bad_alloc const& e)
                {
                    return pika::make_exceptional_future<Result>(e);
                }
                catch (pika::exception_list const& el)
                {
                    return pika::make_exceptional_future<Result>(el);
                }
                catch (...)
                {
                    return pika::make_exceptional_future<Result>(
                        pika::exception_list(std::current_exception()));
                }
            }

            static future<Result> call(future<Result> f)
            {
                PIKA_ASSERT(f.has_exception());
                // Intel complains if this is not explicitly moved
#if defined(PIKA_INTEL_VERSION)
                return PIKA_MOVE(f);
#else
                return f;
#endif
            }

            static future<Result> call(std::exception_ptr const& e)
            {
                try
                {
                    std::rethrow_exception(e);
                }
                catch (std::bad_alloc const&)
                {
                    return pika::make_exceptional_future<Result>(
                        std::current_exception());
                }
                catch (pika::exception_list const& el)
                {
                    return pika::make_exceptional_future<Result>(el);
                }
                catch (...)
                {
                    return pika::make_exceptional_future<Result>(
                        pika::exception_list(std::current_exception()));
                }
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        struct handle_exception_impl<pika::execution::sequenced_task_policy,
            Result> : handle_exception_task_impl<Result>
        {
        };

        template <typename Executor, typename Parameters, typename Result>
        struct handle_exception_impl<
            pika::execution::sequenced_task_policy_shim<Executor, Parameters>,
            Result> : handle_exception_task_impl<Result>
        {
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        struct handle_exception_impl<pika::execution::parallel_task_policy,
            Result> : handle_exception_task_impl<Result>
        {
        };

        template <typename Executor, typename Parameters, typename Result>
        struct handle_exception_impl<
            pika::execution::parallel_task_policy_shim<Executor, Parameters>,
            Result> : handle_exception_task_impl<Result>
        {
        };

#if defined(PIKA_HAVE_DATAPAR)
        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        struct handle_exception_impl<pika::execution::simd_task_policy, Result>
          : handle_exception_task_impl<Result>
        {
        };

        template <typename Result>
        struct handle_exception_impl<pika::execution::par_simd_task_policy,
            Result> : handle_exception_task_impl<Result>
        {
        };
#endif

        using exception_list_termination_handler_type =
            pika::util::function_nonser<void()>;

        PIKA_EXPORT void set_exception_list_termination_handler(
            exception_list_termination_handler_type f);

        PIKA_NORETURN PIKA_EXPORT void exception_list_termination_handler();

        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        struct handle_exception_impl<
            pika::execution::parallel_unsequenced_policy, Result>
        {
            using type = Result;

            PIKA_NORETURN static Result call()
            {
                // any exceptions thrown by algorithms executed with the
                // parallel_unsequenced_policy are to call terminate.
                exception_list_termination_handler();
            }

            PIKA_NORETURN
            static pika::future<Result> call(pika::future<Result>&&)
            {
                exception_list_termination_handler();
            }

            PIKA_NORETURN
            static pika::future<Result> call(std::exception_ptr const&)
            {
                exception_list_termination_handler();
            }
        };

        template <typename Result>
        struct handle_exception_impl<pika::execution::unsequenced_policy, Result>
        {
            using type = Result;

            PIKA_NORETURN static Result call()
            {
                // any exceptions thrown by algorithms executed with the
                // unsequenced_policy are to call terminate.
                exception_list_termination_handler();
            }

            PIKA_NORETURN
            static pika::future<Result> call(pika::future<Result>&&)
            {
                exception_list_termination_handler();
            }

            PIKA_NORETURN
            static pika::future<Result> call(std::exception_ptr const&)
            {
                exception_list_termination_handler();
            }
        };

        template <typename ExPolicy, typename Result = void>
        struct handle_exception
          : handle_exception_impl<std::decay_t<ExPolicy>, Result>
        {
        };
        /// \endcond
    }    // namespace detail

    // we're just reusing our existing implementation
    using pika::exception_list;
}}}    // namespace pika::parallel::v1
