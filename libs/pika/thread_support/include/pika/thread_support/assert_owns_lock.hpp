//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/concepts/has_member_xxx.hpp>
#include <pika/type_support/unused.hpp>

#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace util { namespace detail {
    PIKA_HAS_MEMBER_XXX_TRAIT_DEF(owns_lock)

    template <typename Lock>
    constexpr void assert_owns_lock(Lock const&, int) noexcept
    {
    }

    template <typename Lock>
    constexpr void assert_doesnt_own_lock(Lock const&, int) noexcept
    {
    }

    template <typename Lock>
    typename std::enable_if<has_owns_lock<Lock>::value>::type assert_owns_lock(
        Lock const& l, long) noexcept
    {
        PIKA_ASSERT(l.owns_lock());
        PIKA_UNUSED(l);
    }

    template <typename Lock>
    typename std::enable_if<has_owns_lock<Lock>::value>::type
    assert_doesnt_own_lock(Lock const& l, long) noexcept
    {
        PIKA_ASSERT(!l.owns_lock());
        PIKA_UNUSED(l);
    }
}}}    // namespace pika::util::detail

#define PIKA_ASSERT_OWNS_LOCK(l) ::pika::util::detail::assert_owns_lock(l, 0L)

#define PIKA_ASSERT_DOESNT_OWN_LOCK(l)                                          \
    ::pika::util::detail::assert_doesnt_own_lock(l, 0L)
