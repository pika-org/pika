//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

#include <pika/serialization/exception_ptr.hpp>
#include <pika/serialization/serialization_fwd.hpp>

#include <exception>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace runtime_local { namespace detail {
    PIKA_EXPORT void save_custom_exception(
        pika::serialization::output_archive&, std::exception_ptr const&,
        unsigned int);
    PIKA_EXPORT void load_custom_exception(
        pika::serialization::input_archive&, std::exception_ptr&, unsigned int);
}}}    // namespace pika::runtime_local::detail
