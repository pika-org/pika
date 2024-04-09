//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/allocator_support/internal_allocator.hpp>
#include <pika/modules/logging.hpp>
#include <pika/threading_base/thread_data.hpp>

#include <fmt/format.h>

////////////////////////////////////////////////////////////////////////////////
namespace pika::threads::detail {
    pika::detail::internal_allocator<thread_data_stackful> thread_data_stackful::thread_alloc_;

    thread_data_stackful::~thread_data_stackful()
    {
        PIKA_LOG(debug, "~thread_data_stackful({}), description({}), phase({})", fmt::ptr(this),
            this->get_description(), this->get_thread_phase());
    }
}    // namespace pika::threads::detail
