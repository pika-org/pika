//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/execution_base/receiver.hpp>

namespace pika::execution::experimental::detail {

    template <typename T>
    struct result_type_signature_helper
    {
        using type = pika::execution::experimental::set_value_t(T);
    };

    template <>
    struct result_type_signature_helper<void>
    {
        using type = pika::execution::experimental::set_value_t();
    };

    template <typename T>
    using result_type_signature_helper_t =
        typename result_type_signature_helper<T>::type;

}    // namespace pika::execution::experimental::detail
