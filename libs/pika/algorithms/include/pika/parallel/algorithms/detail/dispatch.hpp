//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/execution/executors/execution.hpp>
#include <pika/executors/exception_list.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/futures/future.hpp>
#include <pika/modules/errors.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/detail/scoped_executor_parameters.hpp>
#include <pika/parallel/util/result_types.hpp>
#include <pika/serialization/serialization_fwd.hpp>

#if defined(PIKA_HAVE_CXX17_STD_EXECUTION_POLICES)
#include <execution>
#endif
#include <string>
#include <type_traits>
#include <utility>

namespace pika { namespace parallel { inline namespace v1 { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct local_algorithm_result
    {
        using type = Result;
    };

    template <typename Result1, typename Result2>
    struct local_algorithm_result<util::in_out_result<Result1, Result2>>
    {
        using type1 = Result1;
        using type2 = Result2;

        using type = util::in_out_result<type1, type2>;
    };

    template <typename Result>
    struct local_algorithm_result<util::min_max_result<Result>>
    {
        using type = util::min_max_result<Result>;
    };

    template <typename Result1, typename Result2, typename Result3>
    struct local_algorithm_result<
        util::in_in_out_result<Result1, Result2, Result3>>
    {
        using type = util::in_in_out_result<Result1, Result2, Result3>;
    };

    template <>
    struct local_algorithm_result<void>
    {
        using type = void;
    };

    template <typename T>
    using local_algorithm_result_t = typename local_algorithm_result<T>::type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename Result = void>
    struct algorithm
    {
    private:
        constexpr Derived const& derived() const noexcept
        {
            return static_cast<Derived const&>(*this);
        }

    public:
        using result_type = Result;
        using local_result_type = local_algorithm_result_t<result_type>;

        explicit constexpr algorithm(char const* const name) noexcept
          : name_(name)
        {
        }

        ///////////////////////////////////////////////////////////////////////
        // this equivalent to sequential execution
        template <typename ExPolicy, typename... Args>
        PIKA_HOST_DEVICE pika::parallel::util::detail::algorithm_result_t<
            ExPolicy, local_result_type>
        operator()(ExPolicy&& policy, Args&&... args) const
        {
#if !defined(__CUDA_ARCH__)
            try
            {
#endif
                using parameters_type =
                    typename std::decay_t<ExPolicy>::executor_parameters_type;
                using executor_type =
                    typename std::decay_t<ExPolicy>::executor_type;

                pika::parallel::util::detail::scoped_executor_parameters_ref<
                    parameters_type, executor_type>
                    scoped_param(policy.parameters(), policy.executor());

                return pika::parallel::util::detail::
                    algorithm_result<ExPolicy, local_result_type>::get(
                        Derived::sequential(PIKA_FORWARD(ExPolicy, policy),
                            PIKA_FORWARD(Args, args)...));
#if !defined(__CUDA_ARCH__)
            }
            catch (...)
            {
                // this does not return
                return pika::parallel::v1::detail::handle_exception<ExPolicy,
                    local_result_type>::call();
            }
#endif
        }

    public:
        ///////////////////////////////////////////////////////////////////////
        // main sequential dispatch entry points

        template <typename ExPolicy, typename... Args>
        constexpr pika::parallel::util::detail::algorithm_result_t<ExPolicy,
            local_result_type>
        call2(ExPolicy&& policy, std::true_type, Args&&... args) const
        {
            using result_handler =
                pika::parallel::util::detail::algorithm_result<ExPolicy,
                    local_result_type>;

            auto exec = policy.executor();    // avoid move after use
            if constexpr (pika::is_async_execution_policy_v<
                              std::decay_t<ExPolicy>>)
            {
                // specialization for all task-based (asynchronous) execution
                // policies

                // run the launched task on the requested executor
                return result_handler::get(execution::async_execute(
                    PIKA_MOVE(exec), derived(), PIKA_FORWARD(ExPolicy, policy),
                    PIKA_FORWARD(Args, args)...));
            }
            else if constexpr (std::is_void_v<local_result_type>)
            {
                execution::sync_execute(PIKA_MOVE(exec), derived(),
                    PIKA_FORWARD(ExPolicy, policy), PIKA_FORWARD(Args, args)...);
                return result_handler::get();
            }
            else
            {
                return result_handler::get(execution::sync_execute(
                    PIKA_MOVE(exec), derived(), PIKA_FORWARD(ExPolicy, policy),
                    PIKA_FORWARD(Args, args)...));
            }
        }

        // main parallel dispatch entry point
        template <typename ExPolicy, typename... Args>
        PIKA_FORCEINLINE static constexpr pika::parallel::util::detail::
            algorithm_result_t<ExPolicy, local_result_type>
            call2(ExPolicy&& policy, std::false_type, Args&&... args)
        {
            return Derived::parallel(
                PIKA_FORWARD(ExPolicy, policy), PIKA_FORWARD(Args, args)...);
        }

        template <typename ExPolicy, typename... Args>
        PIKA_FORCEINLINE constexpr pika::parallel::util::detail::
            algorithm_result_t<ExPolicy, local_result_type>
            call(ExPolicy&& policy, Args&&... args) const
        {
            using is_seq = pika::is_sequenced_execution_policy<ExPolicy>;
            return call2(PIKA_FORWARD(ExPolicy, policy), is_seq(),
                PIKA_FORWARD(Args, args)...);
        }

#if defined(PIKA_HAVE_CXX17_STD_EXECUTION_POLICES)
        // main dispatch entry points for std execution policies
        template <typename... Args>
        PIKA_FORCEINLINE constexpr pika::parallel::util::detail::
            algorithm_result_t<pika::execution::sequenced_policy,
                local_result_type>
            call(std::execution::sequenced_policy, Args&&... args) const
        {
            return call2(pika::execution::seq, std::true_type(),
                PIKA_FORWARD(Args, args)...);
        }

        template <typename... Args>
        PIKA_FORCEINLINE constexpr pika::parallel::util::detail::
            algorithm_result_t<pika::execution::parallel_policy,
                local_result_type>
            call(std::execution::parallel_policy, Args&&... args) const
        {
            return call2(pika::execution::par, std::false_type(),
                PIKA_FORWARD(Args, args)...);
        }

        template <typename... Args>
        PIKA_FORCEINLINE constexpr pika::parallel::util::detail::
            algorithm_result_t<pika::execution::parallel_unsequenced_policy,
                local_result_type>
            call(std::execution::parallel_unsequenced_policy,
                Args&&... args) const
        {
            return call2(pika::execution::par_unseq, std::false_type(),
                PIKA_FORWARD(Args, args)...);
        }

#if defined(PIKA_HAVE_CXX20_STD_EXECUTION_POLICES)
        template <typename... Args>
        PIKA_FORCEINLINE constexpr pika::parallel::util::detail::
            algorithm_result_t<pika::execution::unsequenced_policy,
                local_result_type>
            call(std::execution::unsequenced_policy, Args&&... args) const
        {
            return call2(pika::execution::unseq, std::false_type(),
                PIKA_FORWARD(Args, args)...);
        }
#endif
#endif

    private:
        char const* const name_;

        friend class pika::serialization::access;

        template <typename Archive>
        void serialize(Archive&, unsigned int)
        {
            // no need to serialize 'name_' as it is always initialized by the
            // constructor
        }
    };
}}}}    // namespace pika::parallel::v1::detail
