//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_THREAD_DESCRIPTION)

#include <pika/errors/error_code.hpp>
#include <pika/threading_base/detail/reset_lco_description.hpp>
#include <pika/threading_base/thread_description.hpp>
#include <pika/threading_base/threading_base_fwd.hpp>

namespace pika::threads::detail {
    reset_lco_description::reset_lco_description(thread_id_type const& id,
        ::pika::detail::thread_description const& description, error_code& ec)
      : id_(id)
      , ec_(ec)
    {
        old_desc_ = set_thread_lco_description(id_, description, ec_);
    }

    reset_lco_description::~reset_lco_description()
    {
        set_thread_lco_description(id_, old_desc_, ec_);
    }
}    // namespace pika::threads::detail

#endif    // PIKA_HAVE_THREAD_DESCRIPTION
