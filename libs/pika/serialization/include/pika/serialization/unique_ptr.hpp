//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2014-2015 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/serialization/detail/pointer.hpp>
#include <pika/serialization/serialization_fwd.hpp>
#include <pika/serialization/serialize.hpp>

#include <memory>

namespace pika { namespace serialization {

    template <typename T>
    void load(input_archive& ar, std::unique_ptr<T>& ptr, unsigned)
    {
        detail::serialize_pointer_untracked(ar, ptr);
    }

    template <typename T>
    void save(output_archive& ar, const std::unique_ptr<T>& ptr, unsigned)
    {
        detail::serialize_pointer_untracked(ar, ptr);
    }

    PIKA_SERIALIZATION_SPLIT_FREE_TEMPLATE(
        (template <typename T>), (std::unique_ptr<T>) )
}}    // namespace pika::serialization
