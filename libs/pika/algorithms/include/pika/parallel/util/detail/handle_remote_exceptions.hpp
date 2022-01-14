//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/futures/future.hpp>
#include <pika/modules/errors.hpp>
#include <pika/parallel/util/detail/handle_exception_termination_handler.hpp>

#include <exception>
#include <list>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace parallel { namespace util { namespace detail {

    ///////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct handle_remote_exceptions
    {
        // std::bad_alloc has to be handled separately
        static void call(
            std::exception_ptr const& e, std::list<std::exception_ptr>& errors)
        {
            try
            {
                std::rethrow_exception(e);
            }
            catch (std::bad_alloc const& ba)
            {
                throw ba;
            }
            catch (exception_list const& el)
            {
                for (std::exception_ptr const& ex : el)
                    errors.push_back(ex);
            }
            catch (...)
            {
                errors.push_back(e);
            }
        }

        template <typename T>
        static void call(std::vector<pika::future<T>> const& workitems,
            std::list<std::exception_ptr>& errors)
        {
            for (pika::future<T> const& f : workitems)
            {
                if (f.has_exception())
                    call(f.get_exception_ptr(), errors);
            }

            if (!errors.empty())
                throw exception_list(PIKA_MOVE(errors));
        }

        template <typename T>
        static void call(std::vector<pika::shared_future<T>> const& workitems,
            std::list<std::exception_ptr>& errors)
        {
            for (pika::shared_future<T> const& f : workitems)
            {
                if (f.has_exception())
                    call(f.get_exception_ptr(), errors);
            }

            if (!errors.empty())
                throw exception_list(PIKA_MOVE(errors));
        }
    };

    template <>
    struct handle_remote_exceptions<pika::execution::parallel_unsequenced_policy>
    {
        PIKA_NORETURN static void call(
            std::exception_ptr const&, std::list<std::exception_ptr>&)
        {
            parallel_exception_termination_handler();
        }

        template <typename T>
        static void call(std::vector<pika::future<T>> const& workitems,
            std::list<std::exception_ptr>&)
        {
            for (pika::future<T> const& f : workitems)
            {
                if (f.has_exception())
                    parallel_exception_termination_handler();
            }
        }

        template <typename T>
        static void call(std::vector<pika::shared_future<T>> const& workitems,
            std::list<std::exception_ptr>&)
        {
            for (pika::shared_future<T> const& f : workitems)
            {
                if (f.has_exception())
                    parallel_exception_termination_handler();
            }
        }
    };

    template <>
    struct handle_remote_exceptions<pika::execution::unsequenced_policy>
    {
        PIKA_NORETURN static void call(
            std::exception_ptr const&, std::list<std::exception_ptr>&)
        {
            parallel_exception_termination_handler();
        }

        template <typename T>
        static void call(std::vector<pika::future<T>> const& workitems,
            std::list<std::exception_ptr>&)
        {
            for (pika::future<T> const& f : workitems)
            {
                if (f.has_exception())
                    parallel_exception_termination_handler();
            }
        }

        template <typename T>
        static void call(std::vector<pika::shared_future<T>> const& workitems,
            std::list<std::exception_ptr>&)
        {
            for (pika::shared_future<T> const& f : workitems)
            {
                if (f.has_exception())
                    parallel_exception_termination_handler();
            }
        }
    };
}}}}    // namespace pika::parallel::util::detail
