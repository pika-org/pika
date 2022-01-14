//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/functional/function.hpp>
#include <pika/topology/topology.hpp>

#include <cstddef>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace parallel { namespace execution { namespace detail {
    /// \cond NOINTERNAL
    using get_os_thread_count_type = pika::util::function_nonser<std::size_t()>;
    PIKA_EXPORT void set_get_os_thread_count(get_os_thread_count_type f);
    PIKA_EXPORT std::size_t get_os_thread_count();

    using get_pu_mask_type = pika::util::function_nonser<threads::mask_cref_type(
        threads::topology&, std::size_t)>;
    PIKA_EXPORT void set_get_pu_mask(get_pu_mask_type f);
    PIKA_EXPORT threads::mask_cref_type get_pu_mask(
        threads::topology&, std::size_t);
    /// \endcond
}}}}    // namespace pika::parallel::execution::detail
