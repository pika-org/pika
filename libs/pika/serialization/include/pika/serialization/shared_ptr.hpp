//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2014-2015 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// pikainspect:nodeprecatedinclude:boost/shared_ptr.hpp
// pikainspect:nodeprecatedname:boost::shared_ptr

#pragma once

#include <pika/local/config.hpp>
#include <pika/serialization/detail/pointer.hpp>
#include <pika/serialization/serialization_fwd.hpp>
#include <pika/serialization/serialize.hpp>

#if defined(PIKA_SERIALIZATION_HAVE_BOOST_TYPES)
#include <boost/shared_ptr.hpp>
#endif

#include <memory>

namespace pika { namespace serialization {

#if defined(PIKA_SERIALIZATION_HAVE_BOOST_TYPES)
    template <typename T>
    void load(input_archive& ar, boost::shared_ptr<T>& ptr, unsigned)
    {
        detail::serialize_pointer_tracked(ar, ptr);
    }

    template <typename T>
    void save(output_archive& ar, boost::shared_ptr<T> const& ptr, unsigned)
    {
        detail::serialize_pointer_tracked(ar, ptr);
    }

    PIKA_SERIALIZATION_SPLIT_FREE_TEMPLATE(
        (template <typename T>), (boost::shared_ptr<T>) )
#endif

    template <typename T>
    void load(input_archive& ar, std::shared_ptr<T>& ptr, unsigned)
    {
        detail::serialize_pointer_tracked(ar, ptr);
    }

    template <typename T>
    void save(output_archive& ar, std::shared_ptr<T> const& ptr, unsigned)
    {
        detail::serialize_pointer_tracked(ar, ptr);
    }

    PIKA_SERIALIZATION_SPLIT_FREE_TEMPLATE(
        (template <typename T>), (std::shared_ptr<T>) )
}}    // namespace pika::serialization
