//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_THREAD_DESCRIPTION)

# include <pika/threading_base/thread_description.hpp>
# include <pika/threading_base/threading_base_fwd.hpp>

#if !defined(PIKA_HAVE_MODULE)
# include <pika/errors/error_code.hpp>
#endif

namespace pika::threads::detail {
    struct reset_lco_description
    {
        PIKA_EXPORT reset_lco_description(thread_id_type const& id,
            ::pika::detail::thread_description const& description, error_code& ec = throws);
        PIKA_EXPORT ~reset_lco_description();

        thread_id_type id_;
        ::pika::detail::thread_description old_desc_;
        error_code& ec_;
    };
}    // namespace pika::threads::detail

#endif    // PIKA_HAVE_THREAD_DESCRIPTION
