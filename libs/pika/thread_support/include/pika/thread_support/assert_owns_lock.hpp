//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/concepts/has_member_xxx.hpp>
#include <pika/type_support/unused.hpp>

#include <type_traits>
#endif

///////////////////////////////////////////////////////////////////////////////
namespace pika::util::detail {
#if defined(PIKA_HAVE_MODULE)
    // If we have modules, we also have concepts (or do we?)
    template <typename Lock>
    concept has_owns_lock = requires
    {
        &Lock::owns_lock;
    };
#else
    PIKA_HAS_MEMBER_XXX_TRAIT_DEF(owns_lock)
#endif

    template <typename Lock>
    constexpr void assert_owns_lock(Lock const&, int) noexcept
    {
    }

    template <typename Lock>
    constexpr void assert_doesnt_own_lock(Lock const&, int) noexcept
    {
    }

    template <typename Lock>
    #if defined(PIKA_HAVE_MODULE)
    requires has_owns_lock<Lock> void
    #else
    std::enable_if_t<has_owns_lock<Lock>::value>
    #endif
    assert_owns_lock([[maybe_unused]] Lock const& l, long) noexcept
    {
        PIKA_ASSERT(l.owns_lock());
    }

    template <typename Lock>
    #if defined(PIKA_HAVE_MODULE)
    requires has_owns_lock<Lock> void
    #else
    std::enable_if_t<has_owns_lock<Lock>::value>
    #endif
    assert_doesnt_own_lock([[maybe_unused]] Lock const& l, long) noexcept
    {
        PIKA_ASSERT(!l.owns_lock());
    }
}    // namespace pika::util::detail

#define PIKA_ASSERT_OWNS_LOCK(l) ::pika::util::detail::assert_owns_lock(l, 0L)

#define PIKA_ASSERT_DOESNT_OWN_LOCK(l) ::pika::util::detail::assert_doesnt_own_lock(l, 0L)
