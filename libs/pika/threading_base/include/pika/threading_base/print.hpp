//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/debugging/print.hpp>
#include <pika/threading_base/thread_data.hpp>

#include <iosfwd>

// ------------------------------------------------------------
/// \cond NODETAIL
namespace pika { namespace debug {
    // ------------------------------------------------------------------
    // safely dump thread pointer/description
    // ------------------------------------------------------------------
    template <typename T>
    struct threadinfo;

    // ------------------------------------------------------------------
    // safely dump thread pointer/description
    // ------------------------------------------------------------------
    template <>
    struct threadinfo<threads::thread_data*>
    {
        constexpr threadinfo(threads::thread_data const* v)
          : data(v)
        {
        }

        threads::thread_data const* data;

        PIKA_EXPORT friend std::ostream& operator<<(
            std::ostream& os, threadinfo const& d);
    };

    template <>
    struct threadinfo<threads::thread_id_type*>
    {
        constexpr threadinfo(threads::thread_id_type const* v)
          : data(v)
        {
        }

        threads::thread_id_type const* data;

        PIKA_EXPORT friend std::ostream& operator<<(
            std::ostream& os, threadinfo const& d);
    };

    template <>
    struct threadinfo<threads::thread_id_ref_type*>
    {
        constexpr threadinfo(threads::thread_id_ref_type const* v)
          : data(v)
        {
        }

        threads::thread_id_ref_type const* data;

        PIKA_EXPORT friend std::ostream& operator<<(
            std::ostream& os, threadinfo const& d);
    };

    template <>
    struct threadinfo<pika::threads::thread_init_data>
    {
        constexpr threadinfo(pika::threads::thread_init_data const& v)
          : data(v)
        {
        }

        pika::threads::thread_init_data const& data;

        PIKA_EXPORT friend std::ostream& operator<<(
            std::ostream& os, threadinfo const& d);
    };
}}    // namespace pika::debug
/// \endcond
