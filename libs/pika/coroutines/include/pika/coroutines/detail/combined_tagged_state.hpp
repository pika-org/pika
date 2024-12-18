//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>

#include <cstddef>
#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
namespace pika::threads::detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename T1, typename T2>
    class combined_tagged_state
    {
    private:
        using tagged_state_type = std::int64_t;

        using thread_state_type = std::int8_t;
        using thread_state_ex_type = std::int8_t;
        using tag_type = std::int64_t;

        static std::size_t const state_shift = 56;       // 8th byte
        static std::size_t const state_ex_shift = 48;    // 7th byte

        static tagged_state_type const state_mask = 0xffull;
        static tagged_state_type const state_ex_mask = 0xffull;

        // (1L << 48L) - 1;
        static tagged_state_type const tag_mask = 0x0000'ffff'ffff'ffffull;

        static tag_type extract_tag(tagged_state_type const& i) { return i & tag_mask; }

        static thread_state_type extract_state(tagged_state_type const& i)
        {
            return static_cast<thread_state_type>((i >> state_shift) & state_mask);
        }

        static thread_state_ex_type extract_state_ex(tagged_state_type const& i)
        {
            return static_cast<thread_state_ex_type>((i >> state_ex_shift) & state_ex_mask);
        }

        static tagged_state_type pack_state(T1 state_, T2 state_ex_, tag_type tag)
        {
            tagged_state_type state = static_cast<tagged_state_type>(state_);
            tagged_state_type state_ex = static_cast<tagged_state_type>(state_ex_);

            PIKA_ASSERT(!(state & ~state_mask));
            PIKA_ASSERT(!(state_ex & ~state_ex_mask));
            PIKA_ASSERT(!(state & ~tag_mask));

            return (state << state_shift) | (state_ex << state_ex_shift) | tag;
        }

    public:
        ///////////////////////////////////////////////////////////////////////
        combined_tagged_state() noexcept
          : state_(0)
        {
        }

        combined_tagged_state(T1 state, T2 state_ex, tag_type t = 0)
          : state_(pack_state(state, state_ex, t))
        {
        }

        combined_tagged_state(combined_tagged_state state, tag_type t)
          : state_(pack_state(state.state(), state.state_ex(), t))
        {
        }

        ///////////////////////////////////////////////////////////////////////
        void set(T1 state, T2 state_ex, tag_type t) { state_ = pack_state(state, state_ex, t); }

        ///////////////////////////////////////////////////////////////////////
        bool operator==(combined_tagged_state const& p) const { return state_ == p.state_; }

        bool operator!=(combined_tagged_state const& p) const { return !operator==(p); }

        ///////////////////////////////////////////////////////////////////////
        // state access
        T1 state() const { return static_cast<T1>(extract_state(state_)); }

        void set_state(T1 state) { state_ = pack_state(state, state_ex(), tag()); }

        T2 state_ex() const { return static_cast<T2>(extract_state_ex(state_)); }

        void set_state_ex(T2 state_ex) { state_ = pack_state(state(), state_ex, tag()); }

        ///////////////////////////////////////////////////////////////////////
        // tag access
        tag_type tag() const { return extract_tag(state_); }

        void set_tag(tag_type t) { state_ = pack_state(state(), state_ex(), t); }

    protected:
        tagged_state_type state_;
    };
}    // namespace pika::threads::detail
